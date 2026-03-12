"""
PDF Parser Tool
Extracts text content from PDF files using PyMuPDF (fitz).
Handles both local files and downloaded PDFs.
"""

import io
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logger.warning("PyMuPDF not installed. PDF parsing disabled. Run: pip install PyMuPDF")


class PDFParser:
    """Extract structured text from academic PDFs."""

    def __init__(self, download_dir: Optional[str] = None):
        if fitz is None:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
        self.download_dir = Path(download_dir) if download_dir else Path("./temp_pdfs")
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def parse_local(self, pdf_path: str) -> Dict[str, str]:
        """
        Parse a local PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with 'full_text', 'abstract', 'methodology', 'results', 'references'
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(path))
        return self._extract_sections(doc)

    def parse_url(self, url: str, filename: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Download and parse a PDF from URL.

        Args:
            url: URL to PDF file
            filename: Optional filename for saving

        Returns:
            Dict with extracted sections, or None if download fails
        """
        if not url:
            return None

        try:
            resp = requests.get(url, timeout=60, headers={
                "User-Agent": "Mozilla/5.0 (academic research bot)"
            })
            resp.raise_for_status()

            if "pdf" not in resp.headers.get("content-type", "").lower() and not url.endswith(".pdf"):
                logger.warning(f"URL may not be a PDF: {url}")

            # Parse from memory
            doc = fitz.open(stream=resp.content, filetype="pdf")

            # Optionally save
            if filename:
                save_path = self.download_dir / filename
                save_path.write_bytes(resp.content)
                logger.info(f"PDF saved to {save_path}")

            return self._extract_sections(doc)

        except Exception as e:
            logger.error(f"Failed to download/parse PDF from {url}: {e}")
            return None

    def _extract_sections(self, doc: fitz.Document) -> Dict[str, str]:
        """
        Extract text and attempt to identify key sections.

        Args:
            doc: PyMuPDF document object

        Returns:
            Dict with structured text content
        """
        # Extract full text
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        full_text = "\n".join(pages_text)
        doc.close()

        # Identify sections
        abstract = self._extract_section(full_text, ["abstract"], ["introduction", "keywords", "1."])
        methodology = self._extract_section(
            full_text,
            ["methodology", "method", "methods", "approach", "proposed model", "model description",
             "neural network", "network architecture", "2.", "3."],
            ["results", "experiment", "discussion", "4.", "5."]
        )
        results = self._extract_section(
            full_text,
            ["results", "experiments", "experimental results", "performance", "evaluation"],
            ["conclusion", "discussion", "references", "acknowledgment"]
        )
        conclusion = self._extract_section(
            full_text,
            ["conclusion", "conclusions", "summary"],
            ["references", "acknowledgment", "appendix"]
        )

        return {
            "full_text": full_text[:50000],  # Limit to ~50K chars
            "abstract": abstract[:3000],
            "methodology": methodology[:10000],
            "results": results[:10000],
            "conclusion": conclusion[:5000],
            "num_pages": len(pages_text),
        }

    @staticmethod
    def _extract_section(
        text: str,
        start_keywords: List[str],
        end_keywords: List[str]
    ) -> str:
        """
        Extract a section of text between start and end keywords.

        Args:
            text: Full document text
            start_keywords: Keywords indicating section start
            end_keywords: Keywords indicating section end

        Returns:
            Extracted section text
        """
        text_lower = text.lower()

        # Find the start position
        start_pos = -1
        for kw in start_keywords:
            # Look for section header pattern
            patterns = [
                rf"\n\s*{re.escape(kw)}\s*\n",  # Keyword on its own line
                rf"\n\s*\d+\.?\s*{re.escape(kw)}\s*\n",  # Numbered section
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    start_pos = match.end()
                    break
            if start_pos >= 0:
                break

        if start_pos < 0:
            return ""

        # Find the end position
        end_pos = len(text)
        for kw in end_keywords:
            patterns = [
                rf"\n\s*{re.escape(kw)}\s*\n",
                rf"\n\s*\d+\.?\s*{re.escape(kw)}\s*\n",
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower[start_pos:])
                if match:
                    candidate = start_pos + match.start()
                    if candidate < end_pos:
                        end_pos = candidate
                    break

        return text[start_pos:end_pos].strip()


def parse_pdf(source: str, download_dir: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Parse a PDF from local path or URL.

    Args:
        source: Local file path or URL
        download_dir: Directory for downloaded PDFs

    Returns:
        Dict with extracted sections
    """
    parser = PDFParser(download_dir)

    if source.startswith("http://") or source.startswith("https://"):
        return parser.parse_url(source)
    else:
        return parser.parse_local(source)
