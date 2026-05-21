"""
classic_recommender.py — Daily "classic paper" picks for zotero-arxiv-daily.

Workflow:
  1. Load already-sent classic IDs from state/sent_classics.json (persisted in repo).
  2. Query INSPIRE-HEP for a pool of highly-cited candidate papers.
     pool_mode: 'watchlist_authors' | 'field' | 'both'
  3. Filter out: (a) papers in Zotero corpus, (b) already-sent classics.
  4. Ask the LLM to pick n_picks diverse, relevant papers from the pool.
  5. Return Paper objects; caller must persist the updated sent list after send.

Config (under classic_recommender in custom.yaml):
  enabled: true
  n_picks: 3
  pool_size: 80
  pool_mode: both
  min_citations: 150
  max_paper_age_years: null
  sent_classics_path: state/sent_classics.json   # relative to repo root
  inspire_timeout: 25
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from .protocol import CorpusPaper, Paper

INSPIRE_BASE = "https://inspirehep.net/api/literature"
INSPIRE_FIELDS = "arxiv_eprints,titles,authors,citation_count,abstracts,earliest_date"

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
    "arxiv_id": "<id exactly as given in the candidate list>",
    "reason": "<2-3 sentences: why foundational AND why relevant to the researcher>"
  }},
  ...
]
If there are fewer than {n} suitable candidates, return as many as possible.
"""


# ── Sent-state persistence ────────────────────────────────────────────────────

def _state_path(cfg_path: str) -> str:
    """Resolve path relative to repo root (where the process runs from)."""
    return cfg_path  # workflow runs from repo root via uv run


def load_sent_ids(path: str) -> set[str]:
    try:
        with open(_state_path(path)) as f:
            data = json.load(f)
        ids = set(data.get("sent_ids", []))
        logger.debug(f"Loaded {len(ids)} previously-sent classic IDs from {path}")
        return ids
    except FileNotFoundError:
        logger.info(f"No sent-classics state file at {path}, starting fresh")
        return set()
    except Exception as exc:
        logger.warning(f"Failed to load sent classics state: {exc}")
        return set()


def save_sent_ids(path: str, ids: set[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "sent_ids": sorted(ids),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "count": len(ids),
    }
    try:
        with open(_state_path(path), "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(ids)} sent classic IDs to {path}")
    except Exception as exc:
        logger.warning(f"Failed to save sent classics state: {exc}")


# ── INSPIRE-HEP client ────────────────────────────────────────────────────────

class InspireHEPClient:
    def __init__(self, timeout: int = 25):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def _query(self, q: str, size: int, min_citations: int) -> list[dict]:
        params = {"sort": "mostcited", "size": size, "fields": INSPIRE_FIELDS, "q": q}
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
            if (meta.get("citation_count") or 0) < min_citations:
                continue
            arxiv_ids = [e["value"] for e in meta.get("arxiv_eprints", []) if "value" in e]
            if not arxiv_ids:
                continue
            title = (meta.get("titles") or [{}])[0].get("title", "")
            authors = [a.get("full_name", "") for a in (meta.get("authors") or [])[:6]]
            abstract = (meta.get("abstracts") or [{}])[0].get("value", "")[:500]
            results.append({
                "arxiv_id": arxiv_ids[0],
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "citations": meta.get("citation_count", 0),
                "date": meta.get("earliest_date", ""),
            })
        return results

    def get_author_papers(self, authors: list[str], size: int, min_citations: int) -> list[dict]:
        all_papers: dict[str, dict] = {}
        per_author = max(size // max(len(authors), 1), 15)
        for author in authors:
            surname = author.strip().split()[-1]
            for p in self._query(f"a {surname} and tc hep-th", per_author, min_citations):
                all_papers.setdefault(p["arxiv_id"], p)
            time.sleep(0.3)
        return list(all_papers.values())

    def get_field_papers(self, size: int, min_citations: int) -> list[dict]:
        return self._query("hep-th", size, min_citations)


# ── Filtering helpers ─────────────────────────────────────────────────────────

def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _filter_known(pool: list[dict], corpus: list[CorpusPaper], sent_ids: set[str]) -> list[dict]:
    known_titles = {_norm_title(c.title) for c in corpus}
    before = len(pool)
    filtered = [
        p for p in pool
        if p["arxiv_id"] not in sent_ids
        and _norm_title(p["title"]) not in known_titles
    ]
    logger.debug(
        f"Classic filter: {before} → {len(filtered)} "
        f"(removed {sum(1 for p in pool if p['arxiv_id'] in sent_ids)} already-sent, "
        f"{sum(1 for p in pool if _norm_title(p['title']) in known_titles)} in Zotero)"
    )
    return filtered


def _filter_age(pool: list[dict], max_age_years: Optional[int]) -> list[dict]:
    if not max_age_years:
        return pool
    cutoff = datetime.now(timezone.utc).year - max_age_years
    return [p for p in pool if not p["date"] or int(p["date"][:4]) <= cutoff]


# ── LLM picker ────────────────────────────────────────────────────────────────

def _llm_pick(
    client: OpenAI, model: str,
    pool: list[dict], research_interest: str, n_picks: int,
) -> list[dict]:
    lines = []
    for i, p in enumerate(pool):
        auth = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        lines.append(
            f"[{i+1}] {p['arxiv_id']} | {p['citations']} citations | {p['date'][:4] if p['date'] else '?'}\n"
            f"    Title: {p['title']}\n"
            f"    Authors: {auth}\n"
            f"    Abstract: {p['abstract']}"
        )
    id_map = {p["arxiv_id"]: p for p in pool}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM.format(n=n_picks)},
                {"role": "user", "content": (
                    f"My research interests:\n{research_interest}\n\n"
                    f"Candidate classic papers:\n\n" + "\n\n".join(lines)
                )},
            ],
            max_tokens=800,
            temperature=0.5,  # varied so daily picks differ
        )
        raw = resp.choices[0].message.content.strip()

        # Robust extraction: try after </think>, then inside <think>, then whole raw
        def _try_parse_array(text: str):
            text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE).strip("`").strip()
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            return None

        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        think_text = think_match.group(1) if think_match else ""
        after_think = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        picks = _try_parse_array(after_think) or _try_parse_array(think_text) or _try_parse_array(raw)
        if picks is None:
            raise ValueError(f"No JSON array in response: {raw[:300]}")
        result = []
        for pick in picks[:n_picks]:
            aid = pick.get("arxiv_id", "").strip()
            if aid in id_map:
                result.append(dict(id_map[aid], reason=pick.get("reason", "")))
            else:
                logger.warning(f"LLM returned unknown arXiv ID {aid!r}, skipping")
        return result
    except Exception as exc:
        logger.warning(f"LLM classic pick failed: {exc}")
        # fallback: top-cited
        return [dict(p, reason="") for p in
                sorted(pool, key=lambda x: -x["citations"])[:n_picks]]


# ── Paper conversion ──────────────────────────────────────────────────────────

def _to_paper(p: dict) -> Paper:
    aid = p["arxiv_id"]
    url = f"https://arxiv.org/abs/{aid}"
    pdf_url = f"https://arxiv.org/pdf/{aid}"
    paper = Paper(
        source="classic", title=p["title"], authors=p["authors"],
        abstract=p.get("abstract", ""), url=url, pdf_url=pdf_url,
        score=0.0, tldr=p.get("reason", ""),
    )
    paper.watchlist_hit = None          # type: ignore[attr-defined]
    paper.llm_reason = p.get("reason")  # type: ignore[attr-defined]
    paper.citations = p.get("citations", 0)   # type: ignore[attr-defined]
    paper.is_classic = True             # type: ignore[attr-defined]
    return paper


# ── Main class ────────────────────────────────────────────────────────────────

class ClassicRecommender:
    def __init__(self, config: DictConfig, openai_client: OpenAI):
        self.client = openai_client
        raw = config.get("classic_recommender", {})
        cfg: dict = OmegaConf.to_container(raw, resolve=True) if raw else {}

        self.enabled       = bool(cfg.get("enabled", False))
        self.n_picks       = int(cfg.get("n_picks", 3))
        self.pool_size     = int(cfg.get("pool_size", 80))
        self.pool_mode     = cfg.get("pool_mode", "both")
        self.min_citations = int(cfg.get("min_citations", 150))
        self.max_age_years = cfg.get("max_paper_age_years") or None
        self.state_path    = cfg.get("sent_classics_path", "state/sent_classics.json")
        self.inspire_timeout = int(cfg.get("inspire_timeout", 25))

        llm_r: dict = OmegaConf.to_container(config.reranker.get("llm_reranker", {}), resolve=True) \
            if config.reranker.get("llm_reranker") else {}
        self.research_interest = llm_r.get("research_interest", "theoretical high-energy physics")
        gen = OmegaConf.to_container(config.llm.generation_kwargs, resolve=True)
        self.model = gen.get("model", "deepseek-v4-pro")

        wl: dict = OmegaConf.to_container(config.reranker.get("watchlist", {}), resolve=True) \
            if config.reranker.get("watchlist") else {}
        self.watched_authors: list[str] = wl.get("authors") or []

        self.inspire = InspireHEPClient(timeout=self.inspire_timeout)

    def recommend(self, corpus: list[CorpusPaper]) -> tuple[list[Paper], set[str]]:
        """
        Returns (classic_papers, new_sent_ids_to_persist).
        Caller should call save_sent_ids(path, new_ids) after successful email send.
        """
        if not self.enabled:
            return [], set()

        sent_ids = load_sent_ids(self.state_path)

        # ── Reset if we've exhausted a large fraction of the pool ────────────
        # Hep-th top-500 is a finite set; reset after sending 300 to keep variety
        if len(sent_ids) > 300:
            logger.info("Classic recommender: sent_ids > 300, resetting history for variety")
            sent_ids = set()

        logger.info("Classic recommender: querying INSPIRE-HEP…")
        pool: dict[str, dict] = {}

        if self.pool_mode in ("watchlist_authors", "both") and self.watched_authors:
            for p in self.inspire.get_author_papers(self.watched_authors, self.pool_size, self.min_citations):
                pool.setdefault(p["arxiv_id"], p)
            logger.info(f"  watchlist_authors: {len(pool)} papers")

        if self.pool_mode in ("field", "both"):
            before = len(pool)
            for p in self.inspire.get_field_papers(self.pool_size, self.min_citations):
                pool.setdefault(p["arxiv_id"], p)
            logger.info(f"  field: +{len(pool)-before} papers (total {len(pool)})")

        candidates = list(pool.values())
        candidates = _filter_known(candidates, corpus, sent_ids)
        candidates = _filter_age(candidates, self.max_age_years)

        if not candidates:
            logger.warning("Classic recommender: no candidates after filtering")
            return [], sent_ids

        actual_picks = min(self.n_picks, len(candidates))
        logger.info(f"Classic recommender: LLM picking {actual_picks} from {len(candidates)}…")
        picks = _llm_pick(self.client, self.model, candidates, self.research_interest, actual_picks)

        # Update sent set (only mark as sent after successful pick)
        new_sent = sent_ids | {p["arxiv_id"] for p in picks}
        logger.info(f"Classic recommender: picked {len(picks)} papers")
        return [_to_paper(p) for p in picks], new_sent
