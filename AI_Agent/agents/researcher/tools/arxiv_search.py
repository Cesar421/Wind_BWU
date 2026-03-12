"""
arXiv Search Tool
Searches arXiv for preprints using the official Python arxiv library.
Free, no API key needed. Rate limit: ~3 requests/second.
"""

import logging
from typing import Any, Dict, List, Optional

import arxiv

logger = logging.getLogger(__name__)


class ArxivSearch:
    """Search arXiv for academic preprints."""

    RELEVANT_CATEGORIES = {"cs.LG", "cs.AI", "physics.flu-dyn", "stat.ML", "cs.CE", "eess.SP"}

    def __init__(self, max_results_per_query: int = 50):
        self.max_results = max_results_per_query
        self.client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching a query.

        Args:
            query: arXiv search query (supports boolean operators)
            max_results: Override default max results

        Returns:
            List of normalized paper dicts.
        """
        limit = max_results or self.max_results

        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        try:
            for result in self.client.results(search):
                papers.append(self._normalize(result))
        except Exception as e:
            logger.error(f"arXiv search failed for '{query}': {e}")

        logger.info(f"[arXiv] '{query}' → {len(papers)} results")
        return papers

    def search_by_category(
        self,
        keywords: str,
        categories: Optional[List[str]] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv within specific categories.

        Args:
            keywords: Search keywords
            categories: List of arXiv categories (e.g., ["cs.LG", "physics.flu-dyn"])
            max_results: Max results to return
        """
        if categories is None:
            categories = list(self.RELEVANT_CATEGORIES)

        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        full_query = f"({keywords}) AND ({cat_query})"
        return self.search(full_query, max_results)

    @staticmethod
    def _normalize(result: arxiv.Result) -> Dict[str, Any]:
        """Normalize arxiv.Result to standard schema."""
        # Extract arXiv ID from entry_id URL
        arxiv_id = result.entry_id.split("/abs/")[-1] if result.entry_id else ""

        return {
            "source": "arxiv",
            "source_id": f"arxiv:{arxiv_id}",
            "title": result.title or "",
            "abstract": result.summary or "",
            "year": result.published.year if result.published else None,
            "citation_count": 0,  # arXiv doesn't provide this
            "doi": result.doi or "",
            "arxiv_id": arxiv_id,
            "pdf_url": result.pdf_url or "",
            "url": result.entry_id or "",
            "venue": "arXiv",
            "authors": [a.name for a in (result.authors or [])],
            "categories": [c for c in (result.categories or [])],
        }


def search_arxiv(
    queries: List[str],
    max_results_per_query: int = 50,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function: run multiple queries and aggregate unique results.

    Args:
        queries: List of search queries
        max_results_per_query: Max results per query
        categories: Optional category filter

    Returns:
        Deduplicated list of papers
    """
    searcher = ArxivSearch(max_results_per_query)
    all_papers: Dict[str, Dict] = {}

    for q in queries:
        if categories:
            papers = searcher.search_by_category(q, categories, max_results_per_query)
        else:
            papers = searcher.search(q, max_results_per_query)

        for p in papers:
            pid = p["source_id"]
            if pid and pid not in all_papers:
                all_papers[pid] = p

    logger.info(f"[arXiv] Total unique papers: {len(all_papers)}")
    return list(all_papers.values())
