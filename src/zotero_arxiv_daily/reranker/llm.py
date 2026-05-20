"""
LLM-based reranker with watchlist support.

Registered as reranker name "llm".
Replaces the embedding similarity approach with a direct LLM relevance judgment,
which works much better for small/specialised fields like HEP-th where generic
sentence embeddings have no meaningful resolution.

Config (under reranker in base/custom.yaml):

  watchlist:
    authors: [Witten, Seiberg, ...]         # substring match, case-insensitive
    affiliations: [IAS Princeton, CERN, ...]  # searched in raw full_text header

  llm_reranker:
    research_interest: "..."   # plain-English description of your interests
    score_model: null          # defaults to llm.generation_kwargs.model
    batch_delay: 0.3           # seconds between API calls (rate-limit buffer)
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from ..protocol import CorpusPaper, Paper
from .base import BaseReranker, register_reranker

WATCHLIST_SCORE: float = 100.0

_SYSTEM_PROMPT = """\
You are an expert in theoretical high-energy physics and mathematical physics.
Judge whether a new arXiv preprint is relevant to the researcher's interests.

Respond ONLY with a valid JSON object — no markdown fences, no text outside the JSON:
{"score": <integer 0-10>, "reason": "<one concise sentence>"}

Scoring guide:
  10   – squarely on topic; researcher would almost certainly want to read this
  7-9  – closely related; likely interesting
  4-6  – tangential overlap; might be worth skimming
  1-3  – weak or incidental connection
  0    – completely unrelated
"""


@register_reranker("llm")
class LlmReranker(BaseReranker):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url,
        )

        # llm_reranker sub-config
        raw_llm_cfg = config.reranker.get("llm_reranker", {})
        llm_cfg: dict = OmegaConf.to_container(raw_llm_cfg, resolve=True) if raw_llm_cfg else {}

        self.research_interest: str = llm_cfg.get("research_interest") or (
            "theoretical high-energy physics, quantum field theory, gauge theories, anomalies"
        )

        # score_model: explicit override, else fall back to the main LLM model
        default_model = OmegaConf.to_container(config.llm.generation_kwargs, resolve=True).get(
            "model", "deepseek-v4-pro"
        )
        self.score_model: str = llm_cfg.get("score_model") or default_model
        self.batch_delay: float = float(llm_cfg.get("batch_delay", 0.3))

        # watchlist
        raw_wl = config.reranker.get("watchlist", {})
        wl: dict = OmegaConf.to_container(raw_wl, resolve=True) if raw_wl else {}
        self.watched_authors: list[str] = wl.get("authors") or []
        self.watched_affiliations: list[str] = wl.get("affiliations") or []

    # ── Abstract method — not used in LLM mode ──────────────────────────────
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("LlmReranker does not use get_similarity_score")

    # ── Watchlist helpers ────────────────────────────────────────────────────
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def _check_author_watchlist(self, paper: Paper) -> Optional[dict]:
        if not self.watched_authors:
            return None
        norm_watched = [self._norm(a) for a in self.watched_authors]
        for author in paper.authors:
            an = self._norm(author)
            for wn in norm_watched:
                if wn in an or an in wn:
                    return {"type": "author", "matched": author}
        return None

    def _check_affiliation_watchlist(self, paper: Paper) -> Optional[dict]:
        """
        paper.affiliations is None at rerank time (resolved later by LLM in
        executor.py). As a best-effort heuristic we search the first 8 KB of
        paper.full_text (raw LaTeX / HTML) for the watched affiliation strings.
        """
        if not self.watched_affiliations or not paper.full_text:
            return None
        header = self._norm(paper.full_text[:8000])
        for aff in self.watched_affiliations:
            if self._norm(aff) in header:
                return {"type": "affiliation", "matched": aff}
        return None

    def _check_watchlist(self, paper: Paper) -> Optional[dict]:
        return self._check_author_watchlist(paper) or self._check_affiliation_watchlist(paper)

    # ── LLM scoring ──────────────────────────────────────────────────────────
    def _llm_score(self, paper: Paper) -> tuple[float, str]:
        user_msg = (
            f"My research interests: {self.research_interest}\n\n"
            f"Title: {paper.title}\n\n"
            f"Abstract:\n{paper.abstract}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.score_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip accidental markdown fences
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE).strip("`").strip()
            data = json.loads(raw)
            score = max(0.0, min(10.0, float(data["score"])))
            reason = str(data.get("reason", ""))
            logger.debug(f"LLM [{score:.0f}/10] '{paper.title[:55]}…' — {reason}")
            return score, reason
        except Exception as exc:
            logger.warning(f"LLM scoring failed for '{paper.title[:60]}': {exc}")
            return 0.0, f"scoring error: {exc}"

    # ── Main entry point ─────────────────────────────────────────────────────
    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        logger.info(f"LLM reranker: processing {len(candidates)} papers "
                    f"(model={self.score_model})")

        # 1. Watchlist pass
        for paper in candidates:
            paper.watchlist_hit = self._check_watchlist(paper)  # type: ignore[attr-defined]

        watchlisted = [p for p in candidates if p.watchlist_hit]
        to_score = [p for p in candidates if not p.watchlist_hit]

        if watchlisted:
            labels = [f"{p.watchlist_hit['type']}:{p.watchlist_hit['matched']}" for p in watchlisted]
            logger.info(f"Watchlist pinned {len(watchlisted)} paper(s): {labels}")

        for p in watchlisted:
            p.score = WATCHLIST_SCORE
            p.llm_reason = None  # type: ignore[attr-defined]

        # 2. LLM scoring for the rest
        logger.info(f"LLM scoring {len(to_score)} papers…")
        for paper in to_score:
            score, reason = self._llm_score(paper)
            paper.score = score
            paper.llm_reason = reason  # type: ignore[attr-defined]
            if self.batch_delay > 0:
                time.sleep(self.batch_delay)

        return sorted(candidates, key=lambda p: p.score, reverse=True)
