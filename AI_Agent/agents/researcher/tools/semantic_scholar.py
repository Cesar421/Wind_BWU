"""
Semantic Scholar Search Tool
Searches the Semantic Scholar API for academic papers.
Free API, rate limit: 100 requests / 5 minutes.
"""

import time
import logging
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,year,citationCount,authors,externalIds,url,openAccessPdf,publicationVenue"


class SemanticScholarSearch:
    """Search Semantic Scholar for academic papers on wind pressure ML/DL."""

    def __init__(self, api_key: Optional[str] = None):
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key
        self._last_request_time = 0.0
        self._min_interval = 3.1  # ~100 req / 5 min = 1 req / 3s

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def search(self, query: str, limit: int = 50, year_range: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for papers matching a query.

        Args:
            query: Search string (supports AND/OR operators)
            limit: Max papers to return (max 100 per request)
            year_range: Optional year filter, e.g. "2018-2026"

        Returns:
            List of paper dicts with standardized fields.
        """
        self._rate_limit()

        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": FIELDS,
        }
        if year_range:
            params["year"] = year_range

        try:
            resp = self.session.get(f"{BASE_URL}/paper/search", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Semantic Scholar search failed for '{query}': {e}")
            return []

        papers = data.get("data", [])
        logger.info(f"[Semantic Scholar] '{query}' → {len(papers)} results (total: {data.get('total', '?')})")

        return [self._normalize(p) for p in papers]

    def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed info for a single paper by Semantic Scholar ID."""
        self._rate_limit()
        try:
            resp = self.session.get(
                f"{BASE_URL}/paper/{paper_id}",
                params={"fields": FIELDS + ",references,citations"},
                timeout=30,
            )
            resp.raise_for_status()
            return self._normalize(resp.json())
        except requests.RequestException as e:
            logger.error(f"Failed to fetch paper {paper_id}: {e}")
            return None

    def get_citations(self, paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get papers that cite a given paper."""
        self._rate_limit()
        try:
            resp = self.session.get(
                f"{BASE_URL}/paper/{paper_id}/citations",
                params={"fields": FIELDS, "limit": limit},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            return [self._normalize(item.get("citingPaper", {})) for item in data]
        except requests.RequestException as e:
            logger.error(f"Failed to fetch citations for {paper_id}: {e}")
            return []

    @staticmethod
    def _normalize(paper: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize paper dict to a standard schema."""
        external = paper.get("externalIds", {}) or {}
        pdf_info = paper.get("openAccessPdf") or {}
        venue = paper.get("publicationVenue") or {}
        authors = paper.get("authors", []) or []

        return {
            "source": "semantic_scholar",
            "source_id": paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "year": paper.get("year"),
            "citation_count": paper.get("citationCount", 0),
            "doi": external.get("DOI", ""),
            "arxiv_id": external.get("ArXiv", ""),
            "pdf_url": pdf_info.get("url", ""),
            "url": paper.get("url", ""),
            "venue": venue.get("name", ""),
            "authors": [a.get("name", "") for a in authors],
        }


def search_semantic_scholar(
    queries: List[str],
    api_key: Optional[str] = None,
    limit_per_query: int = 50,
    year_range: str = "2015-2026",
) -> List[Dict[str, Any]]:
    """
    Convenience function: run multiple queries and aggregate results.

    Args:
        queries: List of search query strings
        api_key: Optional API key for higher rate limits
        limit_per_query: Max papers per query
        year_range: Year range filter

    Returns:
        Aggregated list of unique papers (deduplicated by source_id)
    """
    searcher = SemanticScholarSearch(api_key)
    all_papers: Dict[str, Dict] = {}

    for q in queries:
        papers = searcher.search(q, limit=limit_per_query, year_range=year_range)
        for p in papers:
            pid = p["source_id"]
            if pid and pid not in all_papers:
                all_papers[pid] = p

    logger.info(f"[Semantic Scholar] Total unique papers: {len(all_papers)}")
    return list(all_papers.values())
