"""
Paper Analyzer Tool
Uses Claude API to analyze paper content and extract structured model information.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """You are an expert machine learning researcher specializing in time series forecasting and wind engineering. 
Your task is to analyze academic papers and extract structured information about ML/DL models used for predicting wind pressure coefficients (Cp) or similar time series data.

For each paper, extract the following information in JSON format:
{
    "model_type": "category of model (e.g., LSTM, Transformer, CNN, Hybrid, etc.)",
    "architecture_name": "specific architecture name (e.g., Temporal Fusion Transformer, Informer, etc.)",
    "architecture_details": {
        "layers": "description of layer structure",
        "hidden_size": "hidden dimensions if mentioned",
        "attention_mechanism": "type of attention if applicable",
        "special_features": ["list of notable architectural features"]
    },
    "input_features": ["list of input features/variables used"],
    "output_type": "what the model predicts (e.g., single-step, multi-step, probabilistic)",
    "forecast_horizon": "how far ahead the model predicts",
    "metrics_reported": {
        "metric_name": "value",
    },
    "dataset_description": "brief description of training data",
    "dataset_size": "number of samples/timesteps if mentioned",
    "preprocessing": ["list of preprocessing steps"],
    "training_details": {
        "optimizer": "optimizer used",
        "learning_rate": "learning rate if mentioned",
        "epochs": "number of epochs if mentioned",
        "batch_size": "batch size if mentioned"
    },
    "applicable_to_cp_timeseries": true/false,
    "applicability_reasoning": "why this model would/wouldn't work for Cp time series prediction",
    "advantages": ["list of paper-reported advantages"],
    "limitations": ["list of paper-reported or inferred limitations"],
    "key_innovation": "the main contribution or novelty of this paper",
    "relevance_score": 0-10,
    "relevance_reasoning": "why this score was assigned"
}

Focus on extracting actionable information that would help implement these models for wind pressure coefficient (Cp) time series prediction from wind tunnel data (300 pressure taps, ~32768 timesteps at 1000Hz, 11 wind angles).

If information is not available in the paper, use null for that field. Be precise and factual."""

BATCH_RANKING_PROMPT = """You are an expert ML researcher. Given a list of paper abstracts related to ML/DL models for time series prediction, rank them by relevance to this specific task:

**Task**: Predict wind pressure coefficient (Cp) time series on building surfaces without wind tunnel experiments. The data comes from wind tunnel measurements (TPU Aerodynamic Database): 300 pressure taps, ~32768 timesteps at 1000Hz sampling, 11 wind angles (0-50°). Current models (LSTM, Ridge Regression) achieve R²>0.98 for one-step-ahead interpolation but FAIL at true multi-step forecasting (R²<0).

For each paper, provide:
1. relevance_score (0-10): How relevant is this paper to solving the Cp forecasting problem?
2. reasoning: Brief explanation of the score

Return JSON array: [{"index": 0, "relevance_score": 7, "reasoning": "..."}, ...]"""


class PaperAnalyzer:
    """Analyze papers using Claude API to extract model information."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        if not api_key:
            raise ValueError("Se requiere ANTHROPIC_API_KEY. Agrega la key a .env")
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def analyze_paper(self, paper: Dict[str, Any], full_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a single paper using Claude.

        Args:
            paper: Paper dict with at least 'title' and 'abstract'
            full_text: Optional full text or methodology section

        Returns:
            Analysis dict with structured model information
        """
        # Build the content to analyze
        content_parts = [f"**Title**: {paper.get('title', 'Unknown')}"]

        if paper.get("abstract"):
            content_parts.append(f"**Abstract**: {paper['abstract']}")

        if full_text:
            # Truncate to avoid token limits
            truncated = full_text[:15000]
            content_parts.append(f"**Paper Content** (truncated):\n{truncated}")

        if paper.get("venue"):
            content_parts.append(f"**Published in**: {paper['venue']}")
        if paper.get("year"):
            content_parts.append(f"**Year**: {paper['year']}")

        content = "\n\n".join(content_parts)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=ANALYSIS_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"Analyze this paper and extract model information:\n\n{content}"
                }],
            )

            response_text = response.content[0].text

            # Parse JSON from response
            analysis = self._extract_json(response_text)
            if analysis:
                analysis["paper_title"] = paper.get("title", "")
                analysis["paper_source_id"] = paper.get("source_id", "")
                return analysis
            else:
                logger.warning(f"Could not parse analysis for: {paper.get('title', '')}")
                return {"error": "parse_failed", "raw_response": response_text[:500]}

        except Exception as e:
            logger.error(f"Claude analysis failed for '{paper.get('title', '')}': {e}")
            return {"error": str(e)}

    def rank_papers(self, papers: List[Dict[str, Any]], batch_size: int = 20) -> List[Dict[str, Any]]:
        """
        Rank a batch of papers by relevance using Claude.

        Args:
            papers: List of paper dicts
            batch_size: Papers per API call

        Returns:
            Papers with added 'relevance_score' and 'relevance_reasoning' fields
        """
        ranked_papers = list(papers)  # Copy

        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]

            # Format batch for ranking
            batch_text = ""
            for j, p in enumerate(batch):
                batch_text += f"\n---\n**[{j}]** {p.get('title', 'Unknown')}\n"
                batch_text += f"Year: {p.get('year', '?')} | Citations: {p.get('citation_count', 0)}\n"
                abstract = p.get("abstract", "No abstract available")
                batch_text += f"Abstract: {abstract[:500]}\n"

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=BATCH_RANKING_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": f"Rank these {len(batch)} papers:\n{batch_text}"
                    }],
                )

                rankings = self._extract_json(response.content[0].text)
                if isinstance(rankings, list):
                    for r in rankings:
                        idx = r.get("index", -1)
                        if 0 <= idx < len(batch):
                            global_idx = i + idx
                            ranked_papers[global_idx]["relevance_score"] = r.get("relevance_score", 0)
                            ranked_papers[global_idx]["relevance_reasoning"] = r.get("reasoning", "")

            except Exception as e:
                logger.error(f"Batch ranking failed: {e}")
                # Assign default scores
                for j in range(i, min(i + batch_size, len(ranked_papers))):
                    ranked_papers[j].setdefault("relevance_score", 5)

        return ranked_papers

    def generate_search_queries(self, objective: str, existing_queries: List[str]) -> List[str]:
        """
        Use Claude to generate additional search queries.

        Args:
            objective: Research objective description
            existing_queries: Queries already used

        Returns:
            List of new search queries
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": f"""Generate 15 diverse academic search queries for finding papers relevant to this objective:

**Objective**: {objective}

**Queries already used** (do NOT repeat these):
{json.dumps(existing_queries, indent=2)}

Return ONLY a JSON array of query strings. Include queries in different styles:
- Exact phrase searches with quotes
- Boolean combinations (AND/OR)
- Broader queries for related techniques
- Queries targeting specific model architectures
- Queries for wind engineering + ML intersection"""
                }],
            )

            queries = self._extract_json(response.content[0].text)
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str)]

        except Exception as e:
            logger.error(f"Query generation failed: {e}")

        return []

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract JSON from Claude's response text."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown
        import re
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"(\{[\s\S]*\})",
            r"(\[[\s\S]*\])",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        return None
