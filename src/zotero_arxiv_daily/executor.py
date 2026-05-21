from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from .classic_recommender import ClassicRecommender, save_sent_ids
from .reranker.llm import WATCHLIST_SCORE, LlmReranker
from openai import OpenAI
from tqdm import tqdm


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None
    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )
    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")
    return list(patterns)


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
        self.classic_recommender = ClassicRecommender(config, self.openai_client)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']: c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']

        def get_collection_path(col_key: str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']

        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]

    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return

        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")

        reranked_papers = []
        if len(all_papers) > 0:
            max_n = self.config.executor.max_paper_num

            # Stage 1: LLM relevance score only (parallel, no affiliation boost yet)
            logger.info("Stage 1: LLM scoring all papers...")
            stage1 = self.reranker.rerank(all_papers, corpus)

            # Stage 2: affiliation boost on top 2×N candidates only
            # This avoids running affiliation extraction on hundreds of papers.
            candidates = stage1[:max_n * 2]
            logger.info(f"Stage 2: generating affiliations for top {len(candidates)} candidates...")
            for p in tqdm(candidates):
                p.generate_affiliations(self.openai_client, self.config.llm)

            # Re-score with affiliation boost and re-sort
            # Only LlmReranker supports affiliation boost; skip for other rerankers.
            if isinstance(self.reranker, LlmReranker):
                for p in candidates:
                    if p.score < WATCHLIST_SCORE:
                        hit = self.reranker._check_affiliation_boost(p)
                        p.affiliation_hit = hit
                        if hit:
                            p.score = 0.8 * p.score + 2.0
                            logger.debug(f"Affiliation boost → {p.score:.1f} '{p.title[:50]}'")

            reranked_papers = sorted(candidates, key=lambda p: p.score, reverse=True)[:max_n]

            logger.info("Generating TLDR for top papers...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return

        # Classic picks (returns updated sent-ID set to persist after send)
        classic_papers, new_sent_ids = self.classic_recommender.recommend(corpus)

        if classic_papers:
            logger.info("Generating TLDR for classic papers...")
            for p in tqdm(classic_papers):
                p.generate_tldr(self.openai_client, self.config.llm)

        logger.info("Sending email...")
        email_content = render_email(reranked_papers, classic_papers=classic_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")

        # Persist the updated sent-classics state only after successful send
        if new_sent_ids and self.classic_recommender.enabled:
            save_sent_ids(self.classic_recommender.state_path, new_sent_ids)
