from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
from datetime import datetime, timedelta, timezone

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(result_queue: Any, func: Callable[..., T | None], args: tuple[Any, ...]) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None], args: tuple[Any, ...],
    *, timeout: float, operation: str, paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()
    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None
    process.join(5)
    result_queue.close()
    result_queue.join_thread()
    if status == "ok":
        return payload
    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura
    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _arxiv_weekdays_back(days_back: int) -> list[datetime]:
    """
    Return a list of UTC dates (as datetime) covering the last `days_back`
    *arxiv working days* (Mon–Fri).  Skips weekends.
    ArXiv releases Monday papers on Monday evening UTC, so weekday == the
    submission batch we want.
    """
    dates = []
    cursor = datetime.now(timezone.utc)
    while len(dates) < days_back:
        cursor -= timedelta(days=1)
        if cursor.weekday() < 5:   # Mon=0 … Fri=4
            dates.append(cursor)
    return dates


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
        # days_back: how many arxiv working days to fetch. Default 1 (today only).
        self.days_back: int = int(self.config.source.arxiv.get("days_back", 1))

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        categories = self.config.source.arxiv.category
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)

        if self.days_back <= 1:
            return self._retrieve_via_rss(categories, include_cross_list)
        else:
            return self._retrieve_via_search(categories, include_cross_list)

    # ── RSS path (single day, original behaviour) ─────────────────────────
    def _retrieve_via_rss(self, categories, include_cross_list: bool) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = '+'.join(categories)
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")

        allowed_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_types
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        logger.info(f"RSS: {len(all_paper_ids)} paper IDs for {query}")
        return self._fetch_by_ids(client, all_paper_ids)

    # ── Search API path (multi-day batching) ──────────────────────────────
    def _retrieve_via_search(self, categories, include_cross_list: bool) -> list[ArxivResult]:
        """
        Fetch papers submitted in the last self.days_back arxiv working days.
        Uses the arxiv Search API with submittedDate range filter.
        Cross-list: when True, includes papers cross-listed TO these categories
        that were originally submitted to other categories.
        """
        client = arxiv.Client(num_retries=10, delay_seconds=10)

        # Build date range: go back days_back+1 working days to be safe with
        # arxiv's ~midnight UTC cutoff; deduplicate by entry ID.
        working_days = _arxiv_weekdays_back(self.days_back + 1)
        oldest = min(working_days, key=lambda d: d)
        newest = datetime.now(timezone.utc)

        start_str = oldest.strftime("%Y%m%d%H%M")
        end_str   = newest.strftime("%Y%m%d%H%M")

        # Build category query
        if include_cross_list:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        else:
            # Primary category only — we post-filter after fetch
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)

        query = f"({cat_filter}) AND submittedDate:[{start_str} TO {end_str}]"
        max_results = 300 if not self.config.executor.debug else 15

        logger.info(f"Search API: fetching up to {max_results} papers [{start_str}→{end_str}] for {categories}")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers: dict[str, ArxivResult] = {}
        max_batch_retries = 5
        try:
            for attempt in range(max_batch_retries):
                try:
                    for result in client.results(search):
                        papers[result.entry_id] = result
                    break
                except arxiv.HTTPError as exc:
                    if exc.status == 429 and attempt < max_batch_retries - 1:
                        wait = 30 * (attempt + 1)
                        logger.warning(f"arXiv API 429, retry {attempt+1}/{max_batch_retries} in {wait}s")
                        sleep(wait)
                    else:
                        raise
        except Exception as exc:
            logger.warning(f"Search API failed ({exc}), falling back to RSS")
            return self._retrieve_via_rss(categories, include_cross_list)

        # Filter to only primary-category papers if cross_list not wanted
        if not include_cross_list:
            cat_set = set(categories)
            papers = {
                eid: r for eid, r in papers.items()
                if r.primary_category in cat_set
            }

        result_list = list(papers.values())
        logger.info(f"Search API: {len(result_list)} papers after filtering")
        return result_list

    def _fetch_by_ids(self, client, all_paper_ids: list[str]) -> list[ArxivResult]:
        raw_papers = []
        bar = tqdm(total=len(all_paper_ids))
        max_batch_retries = 5
        batch_retry_delay = 30
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
            for attempt in range(max_batch_retries):
                try:
                    batch = list(client.results(search))
                    bar.update(len(batch))
                    raw_papers.extend(batch)
                    break
                except arxiv.HTTPError as exc:
                    if exc.status == 429 and attempt < max_batch_retries - 1:
                        wait = batch_retry_delay * (attempt + 1)
                        logger.warning(f"arXiv API 429 on batch {i//20}, retry {attempt+1}/{max_batch_retries} in {wait}s")
                        sleep(wait)
                    else:
                        raise
            if i + 20 < len(all_paper_ids):
                sleep(3)
        bar.close()
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker, (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT, operation="PDF extraction", paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker, (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT, operation="Tar extraction", paper_title=paper.title,
    )
