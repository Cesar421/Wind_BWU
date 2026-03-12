"""
Configuration loader for the Multi-Agent System.
Loads settings.yaml and .env, provides typed access to all config values.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


class Config:
    """Centralized configuration loaded from settings.yaml + .env."""

    def __init__(self, config_path: Optional[str] = None):
        # Resolve paths relative to AI_Agent/
        self.agent_root = Path(__file__).parent.parent
        self.project_root = self.agent_root.parent  # Wind_ML_TimeSeries/

        # Load .env
        env_path = self.agent_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Try .env.example as fallback warning
            example = self.agent_root / ".env.example"
            if example.exists():
                print("[WARNING] .env not found — copy .env.example to .env and add your API keys.")

        # Load YAML
        if config_path is None:
            config_path = self.agent_root / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg: Dict[str, Any] = yaml.safe_load(f)

    # --- API Keys ---
    @property
    def anthropic_api_key(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def serpapi_key(self) -> str:
        return os.getenv("SERPAPI_KEY", "")

    @property
    def semantic_scholar_api_key(self) -> str:
        return os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    @property
    def claude_model(self) -> str:
        return os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    # --- Data Paths ---
    @property
    def raw_data_dir(self) -> Path:
        return (self.agent_root / self._cfg["data"]["raw_data_dir"]).resolve()

    @property
    def processed_data_dir(self) -> Path:
        return (self.agent_root / self._cfg["data"]["processed_data_dir"]).resolve()

    @property
    def angles(self) -> List[int]:
        return self._cfg["data"]["angles"]

    @property
    def sample_frequency(self) -> int:
        return self._cfg["data"]["sample_frequency"]

    # --- Researcher Config ---
    @property
    def researcher(self) -> Dict[str, Any]:
        return self._cfg["researcher"]

    @property
    def literature_output_dir(self) -> Path:
        p = self.agent_root / self._cfg["researcher"]["output_dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def base_search_queries(self) -> List[str]:
        return self._cfg["researcher"]["base_queries"]

    # --- Modeler Config ---
    @property
    def modeler(self) -> Dict[str, Any]:
        return self._cfg["modeler"]

    @property
    def training_config(self) -> Dict[str, Any]:
        return self._cfg["modeler"]["training"]

    @property
    def seed_models(self) -> List[Dict[str, Any]]:
        return self._cfg["modeler"]["seed_models"]

    @property
    def saved_models_dir(self) -> Path:
        p = (self.agent_root / self._cfg["modeler"]["saved_models_dir"]).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_dir(self) -> Path:
        p = (self.agent_root / self._cfg["modeler"]["results_dir"]).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def generated_code_dir(self) -> Path:
        p = (self.agent_root / self._cfg["modeler"]["generated_code_dir"]).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # --- Orchestrator Config ---
    @property
    def orchestrator(self) -> Dict[str, Any]:
        return self._cfg["orchestrator"]

    # --- MLflow ---
    @property
    def mlflow_tracking_uri(self) -> str:
        return os.getenv("MLFLOW_TRACKING_URI", self._cfg["mlflow"]["tracking_uri"])

    @property
    def mlflow_experiment_name(self) -> str:
        return self._cfg["mlflow"]["experiment_name"]

    # --- Full settings dict ---
    @property
    def settings(self) -> Dict[str, Any]:
        return self._cfg

    # --- Logging ---
    @property
    def log_level(self) -> str:
        return self._cfg["logging"]["level"]

    @property
    def log_file(self) -> Path:
        p = self.agent_root / self._cfg["logging"]["file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# Singleton config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
