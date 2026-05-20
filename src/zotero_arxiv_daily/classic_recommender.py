"""
classic_recommender.py — Daily "classic paper" picks for zotero-arxiv-daily.

Workflow:
  1. Query INSPIRE-HEP for a pool of highly-cited candidate papers.
     Two pool modes (config: classic_recommender.pool_mode):
       'watchlist_authors'  — papers by the authors in reranker.watchlist.authors
       'field'              — broadly highly-cited hep-th papers
       'both'               — union of the above (default)

  2. Filter out papers already present in the user's Zotero corpus
     (matched by arXiv ID or title normalisation).

  3. Ask the LLM to pick `n_picks` papers from the pool that are
       (a) most relevant to research_interest,
       (b) foundational / widely influential,
       (c) diverse across sub-topics.
     The LLM returns structured JSON so we can cross-reference arXiv IDs.

  4. Return a list of Paper objects (score=0, marked as classics) ready
     to be rendered in a separate email section.

Config (added to base.yaml / custom.yaml under key `classic_recommender`):

  classic_recommender:
    enabled: true
    n_picks: 3
    pool_size: 60          # candidates fetched from INSPIRE per query
    pool_mode: both        # 'watchlist_authors' | 'field' | 'both'
    min_citations: 100     # ignore papers below this citation count
    max_paper_age_years: null  # null = no limit; 5 = only papers ≥ 5 years old
    inspire_timeout: 20
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from .protocol import CorpusPaper, Paper

# ─────────────────────────────────────────────────────────────────────────────
INSPIRE_BASE = "https://inspirehep.net/api/literature"
INSPIRE_FIELDS = "arxiv_eprints,titles,authors,citation_count,abstracts,earliest_date"
ARXIV_BASE = "https://export.arxiv.org/abs/"

# ─────────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM = """\
You are an expert in theoretical high-energy physics.
Given a list of classic papers and a researcher's interests, select the papers
most valuable to read, prioritising:
  1. Direct relevance to the stated research interests
  2. Foundational importance (widely cited, opened new directions)
  3. Diversity — cover different aspects, not all from one sub-topic

Respond ONLY with a JSON array of exactly {n} objects, no markdown, no preamble:
[
  {{
    "arxiv_id": "<id as given in the list, e.g. hep-th/9802150 or 2106.12345>",
    "reason": "<2–3 sentences: why this paper is important AND why it connects to the researcher's interests today>"
  }},
  ...
]
"""


class InspireHEPClient:
    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def _query(self, q: str, size: int, min_citations: int) -> list[dict]:
        params = {
            "sort": "mostcited",
            "size": size,
            "fields": INSPIRE_FIELDS,
            "q": q,
        }
        try:
            resp = self.session.get(INSPIRE_BASE, params=params, timeout=self.timeout)
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
        except Exception as exc:
            logger.warning(f"INSPIRE query failed ({q!r}): {exc}")
            return []

        results = []
        for hit in hits:
            meta = hit.get("metadata", {})
            citations = meta.get("citation_count", 0) or 0
            if citations < min_citations:
                continue
            arxiv_ids = [e["value"] for e in meta.get("arxiv_eprints", []) if "value" in e]
            if not arxiv_ids:
                continue
            title = (meta.get("titles") or [{}])[0].get("title", "")
            authors = [a.get("full_name", "") for a in (meta.get("authors") or [])[:6]]
            abstract = (meta.get("abstracts") or [{}])[0].get("value", "")
            date_str = meta.get("earliest_date", "")
            results.append({
                "arxiv_id": arxiv_ids[0],
                "title": title,
                "authors": authors,
                "abstract": abstract[:400],
                "citations": citations,
                "date": date_str,
            })
        return results

    def get_watchlist_author_papers(
        self, authors: list[str], size: int, min_citations: int
    ) -> list[dict]:
        """Fetch highly-cited papers for each watched author and merge."""
        all_papers: dict[str, dict] = {}
        per_author = max(size // max(len(authors), 1), 10)
        for author in authors:
            # INSPIRE author search — last name is sufficient
            surname = author.strip().split()[-1]
            papers = self._query(f"a {surname} and tc 1--hep-th", per_author, min_citations)
            for p in papers:
                all_papers.setdefault(p["arxiv_id"], p)
        return list(all_papers.values())

    def get_field_papers(self, size: int, min_citations: int) -> list[dict]:
        """Fetch broadly highly-cited hep-th papers."""
        return self._query("hep-th", size, min_citations)


# ─────────────────────────────────────────────────────────────────────────────

def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _filter_known(candidates: list[dict], corpus: list[CorpusPaper]) -> list[dict]:
    """Remove papers already in the user's Zotero library."""
    known_titles = {_norm_title(c.title) for c in corpus}
    filtered = [p for p in candidates if _norm_title(p["title"]) not in known_titles]
    logger.debug(f"Classic filter: {len(candidates)} → {len(filtered)} after removing Zotero papers")
    return filtered


def _filter_age(candidates: list[dict], max_age_years: Optional[int]) -> list[dict]:
    if not max_age_years:
        return candidates
    cutoff_year = datetime.now(timezone.utc).year - max_age_years
    def _old_enough(p: dict) -> bool:
        try:
            return int(p["date"][:4]) <= cutoff_year
        except Exception:
            return True  # keep if date unknown
    return [p for p in candidates if _old_enough(p)]


def _llm_pick(
    client: OpenAI,
    candidates: list[dict],
    research_interest: str,
    model: str,
    n_picks: int,
) -> list[dict]:
    """
    Ask the LLM to pick n_picks from the candidate list.
    Returns a list of {'arxiv_id': ..., 'reason': ...}.
    """
    # Build a numbered candidate list for the prompt
    lines = []
    id_map = {}  # index → arxiv_id for verification
    for i, p in enumerate(candidates):
        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += " et al."
        lines.append(
            f"[{i+1}] {p['arxiv_id']} | {p['citations']} citations\n"
            f"    Title: {p['title']}\n"
            f"    Authors: {authors_str}\n"
            f"    Abstract: {p['abstract']}"
        )
        id_map[p["arxiv_id"]] = p

    candidate_text = "\n\n".join(lines)
    user_msg = (
        f"My research interests:\n{research_interest}\n\n"
        f"Candidate classic papers (pick exactly {n_picks}):\n\n"
        f"{candidate_text}"
    )
    system = _LLM_SYSTEM.format(n=n_picks)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=800,
            temperature=0.4,  # slight randomness so daily picks vary
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE).strip("`").strip()
        picks = json.loads(raw)
        # Validate and enrich
        result = []
        for pick in picks[:n_picks]:
            arxiv_id = pick.get("arxiv_id", "").strip()
            if arxiv_id in id_map:
                entry = dict(id_map[arxiv_id])
                entry["reason"] = pick.get("reason", "")
                result.append(entry)
            else:
                logger.warning(f"LLM returned unknown arXiv ID {arxiv_id!r}, skipping")
        return result
    except Exception as exc:
        logger.warning(f"LLM classic pick failed: {exc}")
        # fallback: return highest-cited ones
        return [dict(p, reason="") for p in sorted(candidates, key=lambda x: -x["citations"])[:n_picks]]


def _to_paper(p: dict) -> Paper:
    """Convert a classic-pick dict into a Paper object for email rendering."""
    arxiv_id = p["arxiv_id"]
    # Normalise ID to URL form
    if re.match(r"^\d{4}\.\d+", arxiv_id):
        url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    else:
        # Legacy IDs like hep-th/9802150
        url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    paper = Paper(
        source="classic",
        title=p["title"],
        authors=p["authors"],
        abstract=p.get("abstract", ""),
        url=url,
        pdf_url=pdf_url,
        score=0.0,
        tldr=p.get("reason", ""),
    )
    paper.watchlist_hit = None  # type: ignore[attr-defined]
    paper.llm_reason = p.get("reason")  # type: ignore[attr-defined]
    paper.citations = p.get("citations", 0)  # type: ignore[attr-defined]
    paper.is_classic = True  # type: ignore[attr-defined]
    return paper


# ─────────────────────────────────────────────────────────────────────────────

class ClassicRecommender:
    def __init__(self, config: DictConfig, openai_client: OpenAI):
        self.config = config
        self.client = openai_client

        raw = config.get("classic_recommender", {})
        cfg: dict = OmegaConf.to_container(raw, resolve=True) if raw else {}

        self.enabled: bool = bool(cfg.get("enabled", False))
        self.n_picks: int = int(cfg.get("n_picks", 3))
        self.pool_size: int = int(cfg.get("pool_size", 60))
        self.pool_mode: str = cfg.get("pool_mode", "both")
        self.min_citations: int = int(cfg.get("min_citations", 100))
        self.max_age_years: Optional[int] = cfg.get("max_paper_age_years") or None
        self.inspire_timeout: int = int(cfg.get("inspire_timeout", 20))

        # Get research_interest and model from reranker config
        llm_rcfg = config.reranker.get("llm_reranker", {})
        llm_r: dict = OmegaConf.to_container(llm_rcfg, resolve=True) if llm_rcfg else {}
        self.research_interest: str = llm_r.get("research_interest", "theoretical high-energy physics")

        gen_kwargs = OmegaConf.to_container(config.llm.generation_kwargs, resolve=True)
        self.model: str = gen_kwargs.get("model", "deepseek-v4-pro")

        wl_raw = config.reranker.get("watchlist", {})
        wl: dict = OmegaConf.to_container(wl_raw, resolve=True) if wl_raw else {}
        self.watched_authors: list[str] = wl.get("authors") or []

        self.inspire = InspireHEPClient(timeout=self.inspire_timeout)

    def recommend(self, corpus: list[CorpusPaper]) -> list[Paper]:
        if not self.enabled:
            return []

        logger.info("Classic recommender: building candidate pool…")
        candidates: dict[str, dict] = {}

        if self.pool_mode in ("watchlist_authors", "both") and self.watched_authors:
            papers = self.inspire.get_watchlist_author_papers(
                self.watched_authors, self.pool_size, self.min_citations
            )
            for p in papers:
                candidates.setdefault(p["arxiv_id"], p)
            logger.info(f"  watchlist_authors pool: {len(candidates)} papers")

        if self.pool_mode in ("field", "both"):
            papers = self.inspire.get_field_papers(self.pool_size, self.min_citations)
            before = len(candidates)
            for p in papers:
                candidates.setdefault(p["arxiv_id"], p)
            logger.info(f"  field pool: added {len(candidates)-before} papers")

        pool = list(candidates.values())

        if not pool:
            logger.warning("Classic recommender: empty candidate pool, skipping")
            return []

        pool = _filter_known(pool, corpus)
        pool = _filter_age(pool, self.max_age_years)

        if len(pool) < self.n_picks:
            logger.warning(
                f"Classic recommender: only {len(pool)} candidates after filtering, need {self.n_picks}"
            )
            if not pool:
                return []

        logger.info(f"Classic recommender: LLM picking {self.n_picks} from {len(pool)} candidates…")
        picks = _llm_pick(self.client, pool, self.research_interest, self.model, self.n_picks)

        logger.info(f"Classic recommender: selected {len(picks)} papers")
        return [_to_paper(p) for p in picks]
