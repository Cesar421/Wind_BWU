#!/usr/bin/env python3
"""
Wind Pressure Cp Multi-Agent System — CLI Entry Point

Usage:
    python run.py --mode full           # Full pipeline: research + model
    python run.py --mode research       # Only literature search
    python run.py --mode model          # Only model (uses previous research)
    python run.py --mode seed           # Only seed models (no API needed)
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure AI_Agent root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def setup_logging(log_dir: str = "logs"):
    """Configure structured logging."""
    log_path = ROOT / log_dir
    log_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path / f"run_{timestamp}.log", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def check_environment():
    """Verify API keys and dependencies."""
    from config import get_config

    cfg = get_config()
    critical_issues = []
    warnings = []

    if not cfg.anthropic_api_key:
        critical_issues.append("ANTHROPIC_API_KEY not set (required for code generation)")

    if not cfg.serpapi_key:
        warnings.append("SERPAPI_KEY not set (Google Scholar search disabled)")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logging.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logging.warning("No GPU detected — training will be slow")
    except ImportError:
        critical_issues.append("PyTorch not installed")

    # Check data (BDH .mat files)
    bdh_dir = cfg.project_root / "Data" / "Data_All_The_BDH"
    if bdh_dir.exists():
        mat_files = list(bdh_dir.rglob("*.mat"))
        if not mat_files:
            critical_issues.append(f"No .mat files found in {bdh_dir}")
        else:
            logging.info(f"Data: {len(mat_files)} .mat files found in Data_All_The_BDH")
    else:
        critical_issues.append(f"BDH data directory not found: {bdh_dir}")

    for issue in critical_issues:
        logging.error(f"[ERROR] {issue}")
    for warn in warnings:
        logging.warning(f"[WARN] {warn}")

    return len(critical_issues) == 0


def run_seed_only():
    """Run only seed models (no API keys needed)."""
    logging.info("Seed mode: running base models only...")

    from agents.modeler.agent import run_modeler
    result = run_modeler(model_candidates=[], max_models=4)

    logging.info(f"\nSeed models complete. Evaluated: {len(result.get('evaluated_models', []))}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Wind Pressure Cp — Multi-Agent ML System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "research", "model", "seed"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Max research→model feedback cycles (default: 2)",
    )
    parser.add_argument(
        "--target-r2",
        type=float,
        default=0.3,
        help="Target true forecasting R² to stop (default: 0.3)",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=15,
        help="Max models to implement and test (default: 15)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Log directory (default: logs/)",
    )

    args = parser.parse_args()
    setup_logging(args.log_dir)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Wind Pressure Cp — Multi-Agent ML System")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   Max iterations: {args.max_iterations}")
    logger.info(f"   Target R²: {args.target_r2}")
    logger.info("=" * 60)

    # Check environment
    env_ok = check_environment()
    if not env_ok and args.mode != "seed":
        logger.error("Environment check failed. Fix issues or use --mode seed")
        sys.exit(1)

    # Run
    if args.mode == "seed":
        result = run_seed_only()
    else:
        mode_map = {"full": "full", "research": "research_only", "model": "model_only"}
        from agents.orchestrator import run_pipeline
        result = run_pipeline(
            mode=mode_map[args.mode],
            max_iterations=args.max_iterations,
            target_r2=args.target_r2,
        )

    logger.info("\nPipeline complete.")
    return result


if __name__ == "__main__":
    main()
