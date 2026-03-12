"""
Google Scholar Search Tool (via SerpAPI)
Searches Google Scholar for academic papers.
Requires SerpAPI key (~$50/mo). Essential for wind engineering journals.
"""

import logging
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search"


class GoogleScholarSearch:
    """Search Google Scholar via SerpAPI."""

    def __init__(self, api_key: str):
        if not api_key or api_key == "your-serpapi-key-here":
            raise ValueError(
                "Se requiere una SerpAPI key válida. "
                "Regístrate en https://serpapi.com/ y agrega la key a .env"
            )
        self.api_key = api_key
        self.session = requests.Session()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
    def search(
        self,
        query: str,
        num_results: int = 20,
        year_low: Optional[int] = None,
        year_high: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search Google Scholar for papers.

        Args:
            query: Search string
            num_results: Number of results (max 20 per page)
            year_low: Minimum publication year
            year_high: Maximum publication year

        Returns:
            List of normalized paper dicts.
        """
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": min(num_results, 20),
            "hl": "en",
        }
        if year_low:
            params["as_ylo"] = year_low
        if year_high:
            params["as_yhi"] = year_high

        try:
            resp = self.session.get(SERPAPI_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Google Scholar search failed for '{query}': {e}")
            return []

        results = data.get("organic_results", [])
        logger.info(f"[Google Scholar] '{query}' → {len(results)} results")

        return [self._normalize(r) for r in results]

    def search_paginated(
        self,
        query: str,
        total_results: int = 50,
        year_low: Optional[int] = None,
        year_high: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search with pagination to get more results.

        Args:
            query: Search string
            total_results: Total desired results
            year_low: Min year
            year_high: Max year

        Returns:
            Aggregated list of papers
        """
        all_results = []
        offset = 0
        page_size = 20

        while offset < total_results:
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "num": page_size,
                "start": offset,
                "hl": "en",
            }
            if year_low:
                params["as_ylo"] = year_low
            if year_high:
                params["as_yhi"] = year_high

            try:
                resp = self.session.get(SERPAPI_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("organic_results", [])
                if not results:
                    break
                all_results.extend([self._normalize(r) for r in results])
                offset += page_size
            except requests.RequestException as e:
                logger.error(f"Google Scholar pagination failed at offset {offset}: {e}")
                break

        return all_results

    @staticmethod
    def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize SerpAPI Google Scholar result to standard schema."""
        publication_info = result.get("publication_info", {})
        inline_links = result.get("inline_links", {})
        resources = result.get("resources", [])

        # Try to extract PDF URL from resources
        pdf_url = ""
        for r in resources:
            if r.get("file_format") == "PDF" or "pdf" in r.get("link", "").lower():
                pdf_url = r.get("link", "")
                break

        # Extract authors from publication_info
        authors_str = publication_info.get("summary", "")
        authors = []
        if authors_str:
            # Format: "Author1, Author2 - Journal, Year - Publisher"
            parts = authors_str.split(" - ")
            if parts:
                authors = [a.strip() for a in parts[0].split(",") if a.strip()]

        # Extract venue
        venue = ""
        if len(authors_str.split(" - ")) > 1:
            venue_part = authors_str.split(" - ")[1]
            venue = venue_part.split(",")[0].strip() if venue_part else ""

        # Extract year
        year = None
        year_str = publication_info.get("summary", "")
        import re
        year_match = re.search(r"\b(19|20)\d{2}\b", year_str)
        if year_match:
            year = int(year_match.group())

        cited_by = result.get("inline_links", {}).get("cited_by", {})
        citation_count = cited_by.get("total", 0) if isinstance(cited_by, dict) else 0

        return {
            "source": "google_scholar",
            "source_id": f"gs:{result.get('result_id', '')}",
            "title": result.get("title", ""),
            "abstract": result.get("snippet", ""),
            "year": year,
            "citation_count": citation_count,
            "doi": "",  # Google Scholar doesn't directly provide DOI
            "arxiv_id": "",
            "pdf_url": pdf_url,
            "url": result.get("link", ""),
            "venue": venue,
            "authors": authors[:10],  # Limit to first 10 authors
        }


def search_google_scholar(
    queries: List[str],
    api_key: str,
    results_per_query: int = 20,
    year_low: int = 2015,
    year_high: int = 2026,
) -> List[Dict[str, Any]]:
    """
    Convenience function: run multiple queries and aggregate results.

    Args:
        queries: List of search query strings
        api_key: SerpAPI key
        results_per_query: Results per query
        year_low: Min publication year
        year_high: Max publication year

    Returns:
        Deduplicated list of papers (by title similarity)
    """
    searcher = GoogleScholarSearch(api_key)
    all_papers: List[Dict] = []
    seen_titles: set = set()

    for q in queries:
        papers = searcher.search(q, num_results=results_per_query, year_low=year_low, year_high=year_high)
        for p in papers:
            # Deduplicate by normalized title
            norm_title = p["title"].lower().strip()
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                all_papers.append(p)

    logger.info(f"[Google Scholar] Total unique papers: {len(all_papers)}")
    return all_papers
