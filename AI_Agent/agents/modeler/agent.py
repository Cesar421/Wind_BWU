"""
Modeler Agent
LangGraph StateGraph that orchestrates the full modeling pipeline:
load candidates → prioritize → generate code → validate → train → evaluate → iterate → report
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class ModelerState(TypedDict):
    """State shared across all nodes in the modeler graph."""
    # Inputs
    model_candidates: List[Dict[str, Any]]  # From researcher
    # Pipeline state
    prioritized_models: List[Dict[str, Any]]
    generated_models: List[Dict[str, Any]]  # code + file + validation
    trained_models: List[Dict[str, Any]]    # history + checkpoint
    evaluated_models: List[Dict[str, Any]]  # full metrics
    # Data
    X_train: Optional[np.ndarray]
    y_train: Optional[np.ndarray]
    X_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    X_test: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    test_seq_for_forecast: Optional[np.ndarray]
    y_true_future: Optional[np.ndarray]
    # Control
    current_model_idx: int
    iteration: int
    max_models: int
    errors: List[str]
    status: str


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(state: ModelerState) -> ModelerState:
    """Load and prepare wind pressure data from postprocessed .npy files for training."""
    logger.info("Loading wind pressure data from postprocessed dataset...")

    try:
        # Add AI_Agent root to path
        agent_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(agent_root))

        from config import get_config
        from data_adapter import PostProcessDataAdapter, WindDataAdapter

        cfg = get_config()
        data_cfg = cfg.settings["data"]
        seq_length = cfg.settings["modeler"]["sequence"]["default_length"]
        use_postprocessed = data_cfg.get("use_postprocessed", True)

        if use_postprocessed:
            # --- Postprocessed multivariate data (4 faces) ---
            faces = data_cfg.get("faces", ["windward", "leeward", "sideleft", "sideright"])
            use_all = data_cfg.get("use_all_buildings", True)
            adapter = PostProcessDataAdapter()

            if use_all:
                logger.info("  Loading ALL buildings (multivariate, all alphas)...")
                step = data_cfg.get("sequence_step", 10)
                data = adapter.get_multi_building_data(
                    seq_length=seq_length, step=step, faces=faces, normalize=True,
                )
                logger.info(
                    f"  Loaded {data['buildings_loaded']} buildings: "
                    f"train={data['X_train'].shape}, features={data['num_features']}"
                )
            else:
                alpha = data_cfg.get("default_alpha", "Alpha1_4")
                ratio = data_cfg.get("default_building_ratio", "1_1_3")
                adapter = PostProcessDataAdapter(alpha=alpha, building_ratio=ratio)
                logger.info(f"  Using alpha={alpha}, building_ratio={ratio}, faces={faces}")
                data = adapter.get_multi_angle_data(
                    seq_length=seq_length, faces=faces, normalize=True,
                    alpha=alpha, ratio=ratio,
                )

            state["X_train"] = data["X_train"]
            state["y_train"] = data["y_train"]
            state["X_val"] = data["X_val"]
            state["y_val"] = data["y_val"]
            state["X_test"] = data["X_test"]
            state["y_test"] = data["y_test"]

            # For forecasting evaluation, load one building/angle test seed
            alpha = data_cfg.get("default_alpha", "Alpha1_4")
            ratio = data_cfg.get("default_building_ratio", "1_1_3")
            single_adapter = PostProcessDataAdapter(alpha=alpha, building_ratio=ratio)
            single_data = single_adapter.get_training_data(
                angle=0, seq_length=seq_length, faces=faces, normalize=True,
            )
            state["test_seq_for_forecast"] = single_data["test_seed"]
            state["y_true_future"] = single_data["y_future"]

        else:
            # --- Legacy: raw .mat files (univariate) ---
            alpha = data_cfg.get("default_alpha", "Alpha1_4")
            ratio = data_cfg.get("default_building_ratio", "1_1_3")
            adapter = WindDataAdapter(alpha=alpha, building_ratio=ratio)
            logger.info(f"  Using raw .mat data: alpha={alpha}, building_ratio={ratio}")

            data = adapter.get_training_data(
                angle=0, seq_length=seq_length, normalize=True
            )
            state["X_train"] = data["X_train"]
            state["y_train"] = data["y_train"]
            state["X_val"] = data["X_val"]
            state["y_val"] = data["y_val"]
            state["X_test"] = data["X_test"]
            state["y_test"] = data["y_test"]
            state["test_seq_for_forecast"] = data["test_seed"]
            state["y_true_future"] = data["y_future"]

        logger.info(f"  Train: {state['X_train'].shape}, Val: {state['X_val'].shape}, Test: {state['X_test'].shape}")
        logger.info(f"  Input features: {state['X_train'].shape[2]}")
        logger.info(f"  Forecast seed: {state['test_seq_for_forecast'].shape}, Future: {state['y_true_future'].shape}")
        state["status"] = "data_loaded"

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        traceback.print_exc()
        state["errors"].append(f"Data loading: {e}")
        state["status"] = "error"

    return state


def _create_sequences(data: np.ndarray, seq_length: int):
    """Create input-output pairs from a time series."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    X = np.array(X)
    y = np.array(y)
    # Add feature dimension: (N, seq) -> (N, seq, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


# ---------------------------------------------------------------------------
# Prioritize Models
# ---------------------------------------------------------------------------

def prioritize_models(state: ModelerState) -> ModelerState:
    """Prioritize which models to implement first."""
    logger.info("Prioritizing model candidates...")

    candidates = state.get("model_candidates", [])

    # Add seed models that are always available
    seed_models = [
        {
            "name": "SeedLSTM",
            "category": "recurrent",
            "source": "seed",
            "priority": 10,
            "architecture_details": {"type": "LSTM", "layers": 2, "hidden": 64},
            "key_innovation": "Standard LSTM baseline",
        },
        {
            "name": "SeedBiLSTM",
            "category": "recurrent",
            "source": "seed",
            "priority": 9,
            "architecture_details": {"type": "BiLSTM", "layers": 2, "hidden": 64},
            "key_innovation": "Bidirectional context",
        },
        {
            "name": "SeedTCN",
            "category": "convolutional",
            "source": "seed",
            "priority": 9,
            "architecture_details": {"type": "TCN", "channels": [32, 64, 64]},
            "key_innovation": "Dilated causal convolutions",
        },
        {
            "name": "SeedCNNLSTM",
            "category": "hybrid",
            "source": "seed",
            "priority": 8,
            "architecture_details": {"type": "CNN-LSTM", "filters": 64, "hidden": 64},
            "key_innovation": "CNN feature extraction + LSTM temporal",
        },
    ]

    all_models = seed_models + candidates

    # Sort by priority (higher first), then by citation count or relevance score
    def sort_key(m):
        priority = m.get("priority", 5)
        citations = m.get("citation_count", 0)
        relevance = m.get("relevance_score", 0)
        return (priority, citations, relevance)

    all_models.sort(key=sort_key, reverse=True)

    # Limit to max_models
    max_models = state.get("max_models", 15)
    state["prioritized_models"] = all_models[:max_models]

    logger.info(f"  {len(state['prioritized_models'])} models prioritized:")
    for i, m in enumerate(state["prioritized_models"]):
        logger.info(f"    {i + 1}. {m['name']} ({m['category']}, source: {m.get('source', 'literature')})")

    state["status"] = "models_prioritized"
    return state


# ---------------------------------------------------------------------------
# Generate Model Code
# ---------------------------------------------------------------------------

def generate_and_validate_code(state: ModelerState) -> ModelerState:
    """Generate PyTorch code for each prioritized model."""
    logger.info("Generating code for models...")

    from config import get_config
    from agents.modeler.tools.code_generator import CodeGenerator
    from agents.modeler.templates.model_template import SEED_MODEL_REGISTRY

    cfg = get_config()
    generated = state.get("generated_models", [])

    for model_spec in state["prioritized_models"]:
        name = model_spec["name"]

        # Seed models don't need code generation
        if name in SEED_MODEL_REGISTRY:
            generated.append({
                "name": name,
                "source": "seed",
                "class_name": name,
                "validated": True,
                "spec": model_spec,
            })
            logger.info(f"  {name}: seed model (no code gen needed)")
            continue

        # Generate code via Claude
        try:
            code_gen = CodeGenerator(
                api_key=cfg.anthropic_api_key,
                model=cfg.claude_model,
            )

            code, file_path = code_gen.generate_model_code(model_spec)

            # Validate
            success, msg = code_gen.validate_code(code, file_path)

            if not success:
                logger.warning(f"  {name}: validation failed, attempting fix...")
                fixed_code = code_gen.fix_code(code, msg, model_spec)
                # Save fixed code
                Path(file_path).write_text(fixed_code, encoding="utf-8")
                success, msg = code_gen.validate_code(fixed_code, file_path)

            generated.append({
                "name": name,
                "source": "generated",
                "file_path": file_path,
                "validated": success,
                "validation_msg": msg,
                "spec": model_spec,
            })

            status = "[OK]" if success else "[FAIL]"
            logger.info(f"  {status} {name}: {msg[:80]}")

        except Exception as e:
            logger.error(f"  {name}: code generation error: {e}")
            state["errors"].append(f"Code gen for {name}: {e}")
            generated.append({
                "name": name,
                "source": "generated",
                "validated": False,
                "validation_msg": str(e),
                "spec": model_spec,
            })

    state["generated_models"] = generated
    state["status"] = "code_generated"
    return state


# ---------------------------------------------------------------------------
# Train Models
# ---------------------------------------------------------------------------

def train_models(state: ModelerState) -> ModelerState:
    """Train all validated models."""
    logger.info("Training validated models...")

    from config import get_config
    from agents.modeler.tools.trainer import ModelTrainer
    from agents.modeler.templates.model_template import SEED_MODEL_REGISTRY

    cfg = get_config()
    trainer = ModelTrainer(cfg.training_config)
    save_dir = str(Path(__file__).parent.parent.parent / "model_checkpoints")
    trained = state.get("trained_models", [])

    X_train = state["X_train"]
    y_train = state["y_train"]
    X_val = state["X_val"]
    y_val = state["y_val"]

    for gen_model in state["generated_models"]:
        if not gen_model["validated"]:
            logger.info(f"  {gen_model['name']}: skipping (not validated)")
            continue

        name = gen_model["name"]
        logger.info(f"  Training {name}...")

        try:
            # Instantiate model — input_size = num features (4 for multivariate faces)
            n_features = X_train.shape[2] if X_train.ndim >= 3 else 1
            if gen_model["source"] == "seed":
                model_config = {
                    "input_size": n_features,
                    "output_size": n_features,
                    "seq_length": X_train.shape[1],
                }
                model = SEED_MODEL_REGISTRY[name](model_config)
            else:
                model = _load_generated_model(gen_model["file_path"], X_train.shape)

            if model is None:
                trained.append({"name": name, "success": False, "error": "Model instantiation failed"})
                continue

            # Quick convergence check
            converging, initial_loss = trainer.quick_train(
                model, X_train, y_train, X_val, y_val, epochs=5
            )
            if not converging:
                logger.warning(f"  {name}: not converging in quick check (loss={initial_loss:.4f}), continuing anyway...")

            # Re-instantiate for full training (fresh weights)
            if gen_model["source"] == "seed":
                model_config = {
                    "input_size": n_features,
                    "output_size": n_features,
                    "seq_length": X_train.shape[1],
                }
                model = SEED_MODEL_REGISTRY[name](model_config)
            else:
                model = _load_generated_model(gen_model["file_path"], X_train.shape)

            # Full training
            history = trainer.train(
                model, X_train, y_train, X_val, y_val,
                model_name=name, save_dir=save_dir,
            )

            trained.append({
                "name": name,
                "success": True,
                "history": {k: v for k, v in history.items() if k != "train_loss" and k != "val_loss"},
                "best_val_loss": history["best_val_loss"],
                "training_time": history["training_time_sec"],
                "checkpoint_path": history.get("checkpoint_path"),
                "source": gen_model["source"],
                "file_path": gen_model.get("file_path"),
            })

            logger.info(f"  {name}: val_loss={history['best_val_loss']:.6f}, time={history['training_time_sec']:.1f}s")

        except Exception as e:
            logger.error(f"  {name}: training error: {e}")
            state["errors"].append(f"Training {name}: {e}")
            trained.append({"name": name, "success": False, "error": str(e)})

    state["trained_models"] = trained
    state["status"] = "models_trained"
    return state


def _load_generated_model(file_path: str, data_shape):
    """Dynamically load a generated model from file."""
    import importlib.util
    from agents.modeler.templates.model_template import BaseWindModel

    try:
        spec = importlib.util.spec_from_file_location("gen_model", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseWindModel) and attr is not BaseWindModel:
                config = {
                    "input_size": data_shape[2] if len(data_shape) >= 3 else 1,
                    "output_size": 1,
                    "seq_length": data_shape[1] if len(data_shape) >= 2 else 100,
                }
                return attr(config)

        return None
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Evaluate Models
# ---------------------------------------------------------------------------

def evaluate_models(state: ModelerState) -> ModelerState:
    """Evaluate all trained models (interpolation + true forecasting)."""
    logger.info("Evaluating trained models...")

    import torch
    from agents.modeler.tools.evaluator import ModelEvaluator
    from agents.modeler.templates.model_template import SEED_MODEL_REGISTRY

    evaluator = ModelEvaluator()
    evaluated = state.get("evaluated_models", [])

    X_test = state["X_test"]
    y_test = state["y_test"]
    test_seq = state["test_seq_for_forecast"]
    y_future = state["y_true_future"]

    for trained_model in state["trained_models"]:
        if not trained_model.get("success"):
            continue

        name = trained_model["name"]
        logger.info(f"  Evaluating {name}...")

        try:
            # Load model from checkpoint
            if trained_model.get("checkpoint_path"):
                checkpoint = torch.load(trained_model["checkpoint_path"], weights_only=False)

                if trained_model.get("source") == "seed":
                    model = SEED_MODEL_REGISTRY[name](checkpoint["config"])
                else:
                    model = _load_generated_model(trained_model["file_path"], X_test.shape)

                if model is not None:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    logger.error(f"  Cannot load model {name}")
                    continue
            else:
                continue

            # Interpolation metrics
            interp_metrics = evaluator.evaluate_interpolation(model, X_test, y_test)

            # True forecasting metrics
            forecast_metrics = evaluator.evaluate_true_forecasting(
                model, test_seq, y_future,
                horizons=[10, 50, 100, 500],
            )

            # Compare with baselines
            comparison = evaluator.compare_with_baselines(interp_metrics)

            result = {
                "name": name,
                "source": trained_model.get("source"),
                "interpolation": interp_metrics,
                "forecasting": forecast_metrics,
                "comparison": comparison,
                "training_time": trained_model.get("training_time"),
                "best_val_loss": trained_model.get("best_val_loss"),
            }

            evaluated.append(result)

            # Log key metrics
            r2_interp = interp_metrics.get("r2", 0)
            rmse_val = interp_metrics.get("rmse", 0)
            forecast_r2 = "N/A"
            if "horizon_100" in forecast_metrics:
                forecast_r2 = f"{forecast_metrics['horizon_100'].get('r2', 0):.4f}"

            logger.info(
                f"  {name}: R²_interp={r2_interp:.4f}, RMSE={rmse_val:.6f}, "
                f"R²_forecast_100={forecast_r2}"
            )

        except Exception as e:
            logger.error(f"  {name}: evaluation error: {e}")
            state["errors"].append(f"Evaluation {name}: {e}")

    state["evaluated_models"] = evaluated
    state["status"] = "models_evaluated"
    return state


# ---------------------------------------------------------------------------
# Generate Report
# ---------------------------------------------------------------------------

def generate_report(state: ModelerState) -> ModelerState:
    """Generate a comprehensive comparison report."""
    logger.info("Generating comparison report...")

    output_dir = Path(__file__).parent.parent.parent / "agent_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    evaluated = state.get("evaluated_models", [])
    if not evaluated:
        logger.warning("  No evaluated models to report.")
        state["status"] = "report_empty"
        return state

    # Sort by forecasting R² (horizon_100) — this is the key metric!
    def forecast_r2(m):
        fc = m.get("forecasting", {})
        if "horizon_100" in fc:
            return fc["horizon_100"].get("r2", -999)
        return -999

    evaluated.sort(key=forecast_r2, reverse=True)

    # Build report
    lines = [
        "# Wind Pressure Cp Model Comparison Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        f"- Models evaluated: {len(evaluated)}",
        f"- Errors: {len(state.get('errors', []))}",
        "",
        "## Rankings (by TRUE FORECASTING R² at horizon=100)",
        "",
        "| Rank | Model | R²_interp | RMSE_interp | R²_forecast_100 | RMSE_forecast_100 | Training Time |",
        "|------|-------|-----------|-------------|-----------------|-------------------|---------------|",
    ]

    for rank, model in enumerate(evaluated, 1):
        interp = model.get("interpolation", {})
        fc100 = model.get("forecasting", {}).get("horizon_100", {})
        lines.append(
            f"| {rank} | {model['name']} | "
            f"{interp.get('r2', 'N/A'):.4f} | "
            f"{interp.get('rmse', 'N/A'):.6f} | "
            f"{fc100.get('r2', 'N/A')} | "
            f"{fc100.get('rmse', 'N/A')} | "
            f"{model.get('training_time', 'N/A')}s |"
        )

    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])

    for model in evaluated:
        lines.append(f"### {model['name']}")
        lines.append(f"Source: {model.get('source', 'unknown')}")
        lines.append("")

        # Interpolation table
        interp = model.get("interpolation", {})
        lines.append("**Interpolation (teacher-forced):**")
        for k, v in interp.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        # Forecasting table
        fc = model.get("forecasting", {})
        if fc:
            lines.append("**True Forecasting (autoregressive):**")
            for horizon, metrics in sorted(fc.items()):
                lines.append(f"\n*{horizon}:*")
                for k, v in metrics.items():
                    lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Errors section
    if state.get("errors"):
        lines.append("## Errors Encountered")
        for err in state["errors"]:
            lines.append(f"- {err}")

    # Write report
    report_path = output_dir / f"model_comparison_report_{timestamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Save raw JSON results
    json_path = output_dir / f"evaluation_results_{timestamp}.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = []
    for m in evaluated:
        entry = {}
        for k, v in m.items():
            if isinstance(v, dict):
                entry[k] = {kk: convert(vv) if not isinstance(vv, dict) else {kkk: convert(vvv) for kkk, vvv in vv.items()} for kk, vv in v.items()}
            else:
                entry[k] = convert(v)
        serializable.append(entry)

    json_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

    logger.info(f"  Report: {report_path}")
    logger.info(f"  JSON: {json_path}")

    state["status"] = "report_generated"
    return state


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_modeler_graph():
    """Build the Modeler Agent as a LangGraph StateGraph."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(ModelerState)

    # Add nodes
    graph.add_node("load_data", load_data)
    graph.add_node("prioritize_models", prioritize_models)
    graph.add_node("generate_and_validate_code", generate_and_validate_code)
    graph.add_node("train_models", train_models)
    graph.add_node("evaluate_models", evaluate_models)
    graph.add_node("generate_report", generate_report)

    # Add edges
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "prioritize_models")
    graph.add_edge("prioritize_models", "generate_and_validate_code")
    graph.add_edge("generate_and_validate_code", "train_models")
    graph.add_edge("train_models", "evaluate_models")
    graph.add_edge("evaluate_models", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_modeler(model_candidates: Optional[List[Dict[str, Any]]] = None, max_models: int = 15):
    """
    Run the full modeler pipeline.

    Args:
        model_candidates: List of model specs from researcher agent (can be empty)
        max_models: Max number of models to implement and test
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    initial_state: ModelerState = {
        "model_candidates": model_candidates or [],
        "prioritized_models": [],
        "generated_models": [],
        "trained_models": [],
        "evaluated_models": [],
        "X_train": None,
        "y_train": None,
        "X_val": None,
        "y_val": None,
        "X_test": None,
        "y_test": None,
        "test_seq_for_forecast": None,
        "y_true_future": None,
        "current_model_idx": 0,
        "iteration": 0,
        "max_models": max_models,
        "errors": [],
        "status": "initialized",
    }

    graph = build_modeler_graph()
    result = graph.invoke(initial_state)

    logger.info(f"\nModeler complete. Status: {result['status']}")
    logger.info(f"   Models evaluated: {len(result.get('evaluated_models', []))}")
    logger.info(f"   Errors: {len(result.get('errors', []))}")

    return result


if __name__ == "__main__":
    run_modeler()
