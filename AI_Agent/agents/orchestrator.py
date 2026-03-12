"""
Orchestrator Agent
Coordinates Researcher → Modeler pipeline with feedback loops.
This is the top-level agent that runs the entire multi-agent system.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


class OrchestratorState(TypedDict):
    """Top-level pipeline state."""
    mode: str  # "full", "research_only", "model_only"
    # Researcher outputs
    research_results: Optional[Dict[str, Any]]
    model_candidates: List[Dict[str, Any]]
    # Modeler outputs
    modeling_results: Optional[Dict[str, Any]]
    evaluated_models: List[Dict[str, Any]]
    # Feedback
    iteration: int
    max_iterations: int
    best_forecast_r2: float
    target_forecast_r2: float
    # Control
    errors: List[str]
    status: str


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def run_researcher_node(state: OrchestratorState) -> OrchestratorState:
    """Execute the full researcher pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1: RESEARCHER — Literature search")
    logger.info("=" * 60)

    if state["mode"] == "model_only":
        logger.info("  Mode model_only — skipping research")
        # Load previous results if available
        results_dir = Path(__file__).parent / "literature_results"
        candidates_file = results_dir / "model_candidates.json"
        if candidates_file.exists():
            with open(candidates_file, "r", encoding="utf-8") as f:
                state["model_candidates"] = json.load(f)
            logger.info(f"  Loaded {len(state['model_candidates'])} previous candidates")
        state["status"] = "research_skipped"
        return state

    try:
        from agents.researcher.agent import run_researcher
        result = run_researcher()

        state["research_results"] = {
            "status": result.get("status"),
            "n_papers": len(result.get("all_papers", [])),
            "n_candidates": len(result.get("model_candidates", [])),
        }
        state["model_candidates"] = result.get("model_candidates", [])
        state["status"] = "research_complete"

        logger.info(f"  Papers found: {state['research_results']['n_papers']}")
        logger.info(f"  Model candidates: {len(state['model_candidates'])}")

    except Exception as e:
        logger.error(f"  Researcher error: {e}")
        state["errors"].append(f"Researcher: {e}")
        state["status"] = "research_failed"
        # Continue with empty candidates — seed models will still run

    return state


def run_modeler_node(state: OrchestratorState) -> OrchestratorState:
    """Execute the full modeling pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 2: MODELER — Implementation and evaluation")
    logger.info("=" * 60)

    if state["mode"] == "research_only":
        logger.info("  Mode research_only — skipping modeling")
        state["status"] = "modeling_skipped"
        return state

    try:
        from agents.modeler.agent import run_modeler

        result = run_modeler(
            model_candidates=state["model_candidates"],
            max_models=15,
        )

        state["modeling_results"] = {
            "status": result.get("status"),
            "n_trained": len([m for m in result.get("trained_models", []) if m.get("success")]),
            "n_evaluated": len(result.get("evaluated_models", [])),
            "errors": result.get("errors", []),
        }
        state["evaluated_models"] = result.get("evaluated_models", [])

        # Find best forecasting R²
        best_r2 = -999
        for model in state["evaluated_models"]:
            fc = model.get("forecasting", {})
            if "horizon_100" in fc:
                r2 = fc["horizon_100"].get("r2", -999)
                if r2 > best_r2:
                    best_r2 = r2

        state["best_forecast_r2"] = best_r2
        state["status"] = "modeling_complete"

        logger.info(f"  Best R² forecasting (h=100): {best_r2:.4f}")
        logger.info(f"  Models evaluated: {len(state['evaluated_models'])}")

    except Exception as e:
        logger.error(f"  Modeler error: {e}")
        state["errors"].append(f"Modeler: {e}")
        state["status"] = "modeling_failed"

    return state


def should_iterate(state: OrchestratorState) -> str:
    """
    Decide if another research → model cycle is needed.
    Conditions to continue:
    - best_forecast_r2 < target
    - iteration < max_iterations
    - mode is "full"
    """
    if state["mode"] != "full":
        return "finish"

    if state["iteration"] >= state["max_iterations"]:
        logger.info(f"  Max iterations reached ({state['max_iterations']})")
        return "finish"

    if state["best_forecast_r2"] >= state["target_forecast_r2"]:
        logger.info(f"  Target achieved! R²={state['best_forecast_r2']:.4f} >= {state['target_forecast_r2']:.4f}")
        return "finish"

    logger.info(
        f"  Iteration {state['iteration'] + 1}: R²={state['best_forecast_r2']:.4f} < "
        f"{state['target_forecast_r2']:.4f} — re-searching..."
    )
    state["iteration"] += 1
    return "iterate"


def summarize_results(state: OrchestratorState) -> OrchestratorState:
    """Generate final summary."""
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    output_dir = Path(__file__).parent / "agent_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": state["mode"],
        "iterations": state["iteration"],
        "research": state.get("research_results"),
        "modeling": state.get("modeling_results"),
        "best_forecast_r2": state["best_forecast_r2"],
        "target_r2": state["target_forecast_r2"],
        "target_achieved": state["best_forecast_r2"] >= state["target_forecast_r2"],
        "errors": state.get("errors", []),
        "evaluated_models": [
            {
                "name": m["name"],
                "interp_r2": m.get("interpolation", {}).get("r2"),
                "interp_rmse": m.get("interpolation", {}).get("rmse"),
                "forecast_r2_100": m.get("forecasting", {}).get("horizon_100", {}).get("r2"),
            }
            for m in state.get("evaluated_models", [])
        ],
    }

    # Save summary
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    logger.info(f"  Mode: {state['mode']}")
    logger.info(f"  Iterations: {state['iteration']}")
    logger.info(f"  Best R² forecast: {state['best_forecast_r2']:.4f}")
    logger.info(f"  Target R²: {state['target_forecast_r2']:.4f}")
    logger.info(f"  Target achieved: {'YES' if summary['target_achieved'] else 'NO'}")
    logger.info(f"  Total errors: {len(state.get('errors', []))}")

    if state.get("evaluated_models"):
        logger.info("\n  Top models:")
        sorted_models = sorted(
            state["evaluated_models"],
            key=lambda m: m.get("forecasting", {}).get("horizon_100", {}).get("r2", -999),
            reverse=True,
        )
        for i, m in enumerate(sorted_models[:5], 1):
            fc_r2 = m.get("forecasting", {}).get("horizon_100", {}).get("r2", "N/A")
            int_r2 = m.get("interpolation", {}).get("r2", "N/A")
            logger.info(f"    {i}. {m['name']}: R²_forecast={fc_r2}, R²_interp={int_r2}")

    logger.info(f"\n  Results: {summary_path}")
    state["status"] = "completed"
    return state


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

def build_orchestrator_graph():
    """Build the orchestrator as a LangGraph StateGraph."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(OrchestratorState)

    graph.add_node("researcher", run_researcher_node)
    graph.add_node("modeler", run_modeler_node)
    graph.add_node("summarize", summarize_results)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "modeler")
    graph.add_conditional_edges(
        "modeler",
        should_iterate,
        {
            "iterate": "researcher",
            "finish": "summarize",
        },
    )
    graph.add_edge("summarize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    mode: str = "full",
    max_iterations: int = 2,
    target_r2: float = 0.3,
) -> Dict[str, Any]:
    """
    Run the full multi-agent pipeline.

    Args:
        mode: "full" (researcher + modeler), "research_only", or "model_only"
        max_iterations: Max research → model feedback cycles
        target_r2: Target true forecasting R² to stop iterating

    Returns:
        Final pipeline state
    """
    initial_state: OrchestratorState = {
        "mode": mode,
        "research_results": None,
        "model_candidates": [],
        "modeling_results": None,
        "evaluated_models": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "best_forecast_r2": -999.0,
        "target_forecast_r2": target_r2,
        "errors": [],
        "status": "initialized",
    }

    graph = build_orchestrator_graph()
    result = graph.invoke(initial_state)
    return result
