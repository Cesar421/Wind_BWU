"""
Researcher Agent — LangGraph-based literature search and model candidate extraction.

This agent autonomously:
1. Generates diverse search queries
2. Searches Semantic Scholar, arXiv, and Google Scholar in parallel
3. Deduplicates results
4. Ranks papers by relevance using Claude
5. Deep-analyzes top papers (PDF when available)
6. Synthesizes a ranked list of model candidates
7. Saves structured output for the Modeler Agent
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import yaml

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph.graph import END, StateGraph

from config import get_config
from agents.researcher.tools.semantic_scholar import search_semantic_scholar
from agents.researcher.tools.arxiv_search import search_arxiv
from agents.researcher.tools.google_scholar import search_google_scholar
from agents.researcher.tools.pdf_parser import PDFParser
from agents.researcher.tools.paper_analyzer import PaperAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class ResearcherState(TypedDict):
    """State passed between LangGraph nodes."""
    queries: List[str]
    all_papers: List[Dict[str, Any]]
    ranked_papers: List[Dict[str, Any]]
    analyzed_papers: List[Dict[str, Any]]
    model_candidates: List[Dict[str, Any]]
    search_iteration: int
    errors: List[str]
    status: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------
def generate_queries(state: ResearcherState) -> Dict[str, Any]:
    """Generate search queries from config + Claude-generated expansions."""
    config = get_config()
    logger.info("Generating search queries...")

    # Load base queries from config
    queries = list(config.base_search_queries)

    # Load category-based queries from YAML
    queries_yaml_path = Path(__file__).parent / "prompts" / "search_queries.yaml"
    if queries_yaml_path.exists():
        with open(queries_yaml_path, "r", encoding="utf-8") as f:
            query_data = yaml.safe_load(f)
        for category, cat_queries in query_data.get("categories", {}).items():
            queries.extend(cat_queries)

    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)

    # Ask Claude to generate additional queries if API key available
    if config.anthropic_api_key:
        try:
            analyzer = PaperAnalyzer(config.anthropic_api_key, config.claude_model)
            objective = query_data.get("objective", "Find ML/DL models for Cp time series prediction")
            extra = analyzer.generate_search_queries(objective, unique_queries[:15])
            for q in extra:
                q_lower = q.lower().strip()
                if q_lower not in seen:
                    seen.add(q_lower)
                    unique_queries.append(q)
            logger.info(f"  Claude generated {len(extra)} additional queries")
        except Exception as e:
            logger.warning(f"  Could not generate queries with Claude: {e}")

    logger.info(f"  Total queries: {len(unique_queries)}")
    return {"queries": unique_queries, "search_iteration": state.get("search_iteration", 0) + 1}


def search_papers(state: ResearcherState) -> Dict[str, Any]:
    """Execute searches across all sources."""
    config = get_config()
    queries = state["queries"]
    all_papers: List[Dict[str, Any]] = list(state.get("all_papers", []))
    errors: List[str] = list(state.get("errors", []))

    # --- Semantic Scholar ---
    logger.info(f"Searching Semantic Scholar ({len(queries)} queries)...")
    try:
        ss_papers = search_semantic_scholar(
            queries=queries[:30],  # Limit to avoid rate limit
            api_key=config.semantic_scholar_api_key or None,
            limit_per_query=config.researcher.get("max_papers_per_query", 50),
            year_range="2015-2026",
        )
        all_papers.extend(ss_papers)
        logger.info(f"  Semantic Scholar: {len(ss_papers)} papers found")
    except Exception as e:
        errors.append(f"Semantic Scholar error: {e}")
        logger.error(f"  Semantic Scholar error: {e}")

    # --- arXiv ---
    logger.info("Searching arXiv...")
    try:
        arxiv_papers = search_arxiv(
            queries=queries[:20],
            max_results_per_query=30,
            categories=config.researcher.get("sources", {}).get("arxiv", {}).get("categories"),
        )
        all_papers.extend(arxiv_papers)
        logger.info(f"  arXiv: {len(arxiv_papers)} papers found")
    except Exception as e:
        errors.append(f"arXiv error: {e}")
        logger.error(f"  arXiv error: {e}")

    # --- Google Scholar ---
    gs_config = config.researcher.get("sources", {}).get("google_scholar", {})
    if gs_config.get("enabled") and config.serpapi_key:
        logger.info("Searching Google Scholar (SerpAPI)...")
        try:
            gs_papers = search_google_scholar(
                queries=queries[:gs_config.get("max_searches_per_run", 50)],
                api_key=config.serpapi_key,
                results_per_query=20,
                year_low=2015,
                year_high=2026,
            )
            all_papers.extend(gs_papers)
            logger.info(f"  Google Scholar: {len(gs_papers)} papers found")
        except Exception as e:
            errors.append(f"Google Scholar error: {e}")
            logger.error(f"  Google Scholar error: {e}")
    else:
        logger.info("Google Scholar disabled (no SerpAPI key)")

    logger.info(f"Total papers before deduplication: {len(all_papers)}")
    return {"all_papers": all_papers, "errors": errors}


def deduplicate(state: ResearcherState) -> Dict[str, Any]:
    """Remove duplicate papers by DOI and title similarity."""
    papers = state["all_papers"]
    logger.info(f"Deduplicating {len(papers)} papers...")

    seen_dois: set = set()
    seen_titles: set = set()
    unique: List[Dict[str, Any]] = []

    for p in papers:
        # Check DOI
        doi = (p.get("doi") or "").strip().lower()
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)

        # Check title similarity (normalized)
        title = (p.get("title") or "").strip().lower()
        title_normalized = "".join(c for c in title if c.isalnum() or c == " ")
        if title_normalized and title_normalized in seen_titles:
            continue
        if title_normalized:
            seen_titles.add(title_normalized)

        unique.append(p)

    logger.info(f"  After deduplication: {len(unique)} unique papers")
    return {"all_papers": unique}


def rank_relevance(state: ResearcherState) -> Dict[str, Any]:
    """Use Claude to rank papers by relevance."""
    config = get_config()
    papers = state["all_papers"]
    logger.info(f"Ranking {len(papers)} papers by relevance...")

    if not config.anthropic_api_key:
        logger.warning("  No Anthropic API key — assigning scores by heuristics")
        for p in papers:
            score = 5
            title_lower = (p.get("title") or "").lower()
            abstract_lower = (p.get("abstract") or "").lower()
            combined = title_lower + " " + abstract_lower
            # Heuristic scoring
            if "wind pressure" in combined or "cp " in combined or "pressure coefficient" in combined:
                score += 3
            if "time series" in combined or "forecasting" in combined:
                score += 1
            if "deep learning" in combined or "neural network" in combined or "lstm" in combined:
                score += 1
            p["relevance_score"] = min(score, 10)
            p["relevance_reasoning"] = "heuristic scoring (no API key)"
        ranked = sorted(papers, key=lambda x: x.get("relevance_score", 0), reverse=True)
        return {"ranked_papers": ranked}

    # Use Claude for ranking
    try:
        analyzer = PaperAnalyzer(config.anthropic_api_key, config.claude_model)
        ranked = analyzer.rank_papers(papers)
        ranked.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        logger.info(f"  Ranking complete. Top paper: '{ranked[0].get('title', '')[:80]}' (score: {ranked[0].get('relevance_score', '?')})")
    except Exception as e:
        logger.error(f"  Ranking error: {e}")
        ranked = papers  # Fall back to unranked

    return {"ranked_papers": ranked}


def deep_analyze(state: ResearcherState) -> Dict[str, Any]:
    """Download PDFs and perform deep analysis on top papers."""
    config = get_config()
    ranked = state["ranked_papers"]
    top_n = config.researcher.get("top_papers_for_deep_analysis", 30)
    min_score = config.researcher.get("min_relevance_score", 5)

    # Filter top papers
    top_papers = [p for p in ranked if p.get("relevance_score", 0) >= min_score][:top_n]
    logger.info(f"Deep-analyzing {len(top_papers)} papers (score >= {min_score})...")

    if not config.anthropic_api_key:
        logger.warning("  No API key — skipping deep analysis")
        return {"analyzed_papers": top_papers}

    analyzer = PaperAnalyzer(config.anthropic_api_key, config.claude_model)
    pdf_parser = PDFParser(str(config.literature_output_dir / "pdfs"))
    analyzed = []

    for i, paper in enumerate(top_papers):
        logger.info(f"  [{i+1}/{len(top_papers)}] Analyzing: {paper.get('title', '')[:70]}...")

        # Try to get PDF content
        full_text = None
        pdf_url = paper.get("pdf_url", "")
        if pdf_url:
            try:
                pdf_content = pdf_parser.parse_url(pdf_url)
                if pdf_content:
                    full_text = pdf_content.get("methodology", "") + "\n" + pdf_content.get("results", "")
            except Exception as e:
                logger.debug(f"    PDF download failed: {e}")

        # Analyze with Claude
        try:
            analysis = analyzer.analyze_paper(paper, full_text)
            paper["analysis"] = analysis
        except Exception as e:
            logger.warning(f"    Analysis failed: {e}")
            paper["analysis"] = {"error": str(e)}

        analyzed.append(paper)

    return {"analyzed_papers": analyzed}


def synthesize(state: ResearcherState) -> Dict[str, Any]:
    """Synthesize model candidates from analyzed papers."""
    analyzed = state["analyzed_papers"]
    logger.info("Synthesizing model candidates...")

    # Extract unique models from analyses
    model_map: Dict[str, Dict[str, Any]] = {}

    for paper in analyzed:
        analysis = paper.get("analysis", {})
        if "error" in analysis:
            continue

        arch_name = analysis.get("architecture_name", "")
        model_type = analysis.get("model_type", "")
        if not arch_name and not model_type:
            continue

        key = (arch_name or model_type).lower().strip()
        if key not in model_map:
            model_map[key] = {
                "name": arch_name or model_type,
                "category": model_type,
                "papers": [],
                "avg_relevance": 0,
                "architecture_details": analysis.get("architecture_details", {}),
                "applicable_to_cp": analysis.get("applicable_to_cp_timeseries", True),
                "advantages": analysis.get("advantages", []),
                "limitations": analysis.get("limitations", []),
                "key_innovation": analysis.get("key_innovation", ""),
                "best_metrics": analysis.get("metrics_reported", {}),
                "output_type": analysis.get("output_type", ""),
                "forecast_horizon": analysis.get("forecast_horizon", ""),
            }

        model_map[key]["papers"].append({
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "relevance_score": paper.get("relevance_score", 0),
            "source_id": paper.get("source_id", ""),
        })

    # Calculate priority score
    candidates = []
    for key, model in model_map.items():
        paper_count = len(model["papers"])
        avg_score = sum(p["relevance_score"] for p in model["papers"]) / max(paper_count, 1)
        model["paper_count"] = paper_count
        model["avg_relevance"] = round(avg_score, 2)
        model["priority_score"] = round(paper_count * 0.3 + avg_score * 0.7, 2)
        model["high_priority"] = paper_count >= 3
        candidates.append(model)

    # Sort by priority
    candidates.sort(key=lambda x: x["priority_score"], reverse=True)

    logger.info(f"  Model candidates found: {len(candidates)}")
    for i, c in enumerate(candidates[:10]):
        logger.info(f"    {i+1}. {c['name']} (papers: {c['paper_count']}, score: {c['priority_score']})")

    return {"model_candidates": candidates, "status": "completed"}


def save_results(state: ResearcherState) -> Dict[str, Any]:
    """Save all results to disk."""
    config = get_config()
    output_dir = config.literature_output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all papers
    papers_path = output_dir / config.researcher["papers_file"]
    papers_data = {
        "timestamp": timestamp,
        "total_papers": len(state.get("ranked_papers", [])),
        "search_iterations": state.get("search_iteration", 0),
        "papers": state.get("ranked_papers", []),
    }
    with open(papers_path, "w", encoding="utf-8") as f:
        json.dump(papers_data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Papers saved to: {papers_path}")

    # Save model candidates
    candidates_path = output_dir / config.researcher["candidates_file"]
    candidates_data = {
        "timestamp": timestamp,
        "total_candidates": len(state.get("model_candidates", [])),
        "candidates": state.get("model_candidates", []),
    }
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidates_data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Candidates saved to: {candidates_path}")

    # Save synthesis markdown
    synthesis_path = output_dir / config.researcher.get("synthesis_file", "literature_synthesis.md")
    with open(synthesis_path, "w", encoding="utf-8") as f:
        f.write(f"# Literature Synthesis — Wind Pressure Cp Time Series Modeling\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"## Summary\n")
        f.write(f"- Total papers found: {len(state.get('ranked_papers', []))}\n")
        f.write(f"- Papers analyzed in depth: {len(state.get('analyzed_papers', []))}\n")
        f.write(f"- Model candidates identified: {len(state.get('model_candidates', []))}\n\n")

        f.write(f"## Model Candidates (Ranked by Priority)\n\n")
        for i, c in enumerate(state.get("model_candidates", [])):
            f.write(f"### {i+1}. {c['name']}\n")
            f.write(f"- **Category**: {c.get('category', 'N/A')}\n")
            f.write(f"- **Priority Score**: {c.get('priority_score', 'N/A')}\n")
            f.write(f"- **Papers Found**: {c.get('paper_count', 0)}\n")
            f.write(f"- **Key Innovation**: {c.get('key_innovation', 'N/A')}\n")
            if c.get("advantages"):
                f.write(f"- **Advantages**: {', '.join(c['advantages'][:5])}\n")
            if c.get("limitations"):
                f.write(f"- **Limitations**: {', '.join(c['limitations'][:5])}\n")
            f.write("\n")

    logger.info(f"Synthesis saved to: {synthesis_path}")

    return {"status": "saved"}


# ---------------------------------------------------------------------------
# Conditional edge: should we re-search?
# ---------------------------------------------------------------------------
def should_research_more(state: ResearcherState) -> str:
    """Decide if we need to search for more papers."""
    config = get_config()
    min_papers = config.researcher.get("min_papers_threshold", 15)
    max_iterations = config.orchestrator.get("max_research_iterations", 3)

    unique_papers = len(state.get("all_papers", []))
    iteration = state.get("search_iteration", 0)

    if unique_papers < min_papers and iteration < max_iterations:
        logger.info(f"Only {unique_papers} papers found (minimum: {min_papers}). Re-searching...")
        return "research_more"
    return "continue"


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------
def build_researcher_graph() -> StateGraph:
    """Build the Researcher Agent LangGraph workflow."""

    workflow = StateGraph(ResearcherState)

    # Add nodes
    workflow.add_node("generate_queries", generate_queries)
    workflow.add_node("search_papers", search_papers)
    workflow.add_node("deduplicate", deduplicate)
    workflow.add_node("rank_relevance", rank_relevance)
    workflow.add_node("deep_analyze", deep_analyze)
    workflow.add_node("synthesize", synthesize)
    workflow.add_node("save_results", save_results)

    # Define edges
    workflow.set_entry_point("generate_queries")
    workflow.add_edge("generate_queries", "search_papers")
    workflow.add_edge("search_papers", "deduplicate")

    # Conditional: do we have enough papers?
    workflow.add_conditional_edges(
        "deduplicate",
        should_research_more,
        {
            "research_more": "generate_queries",
            "continue": "rank_relevance",
        },
    )

    workflow.add_edge("rank_relevance", "deep_analyze")
    workflow.add_edge("deep_analyze", "synthesize")
    workflow.add_edge("synthesize", "save_results")
    workflow.add_edge("save_results", END)

    return workflow


def run_researcher() -> Dict[str, Any]:
    """Run the Researcher Agent and return final state."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("RESEARCHER AGENT — Starting literature search")
    logger.info("=" * 60)

    graph = build_researcher_graph()
    app = graph.compile()

    initial_state: ResearcherState = {
        "queries": [],
        "all_papers": [],
        "ranked_papers": [],
        "analyzed_papers": [],
        "model_candidates": [],
        "search_iteration": 0,
        "errors": [],
        "status": "starting",
    }

    final_state = app.invoke(initial_state)

    logger.info("=" * 60)
    logger.info(f"Search complete. {len(final_state.get('model_candidates', []))} model candidates.")
    logger.info("=" * 60)

    return final_state


if __name__ == "__main__":
    run_researcher()
