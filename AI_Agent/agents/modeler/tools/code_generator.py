"""
Code Generator Tool
Uses Claude API to generate PyTorch model code from model specifications.
Validates generated code compiles and produces correct output shapes.
"""

import importlib.util
import logging
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from anthropic import Anthropic

logger = logging.getLogger(__name__)

CODE_GEN_SYSTEM_PROMPT = """You are an expert PyTorch developer specializing in time series forecasting models.
Your task is to generate a complete, self-contained PyTorch model class that inherits from BaseWindModel.

CRITICAL REQUIREMENTS:
1. The model class MUST inherit from BaseWindModel
2. The __init__ method MUST call super().__init__(config) 
3. The forward method MUST accept x of shape (batch_size, seq_length, input_size) and return (batch_size, output_size)
4. Use only standard PyTorch modules (nn.Module subclasses)
5. The code must be self-contained — all imports at the top
6. Include docstrings explaining the architecture
7. Handle variable seq_length gracefully (don't hardcode dimensions)

TEMPLATE:
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

# Import base class
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from templates.model_template import BaseWindModel


class {ModelName}(BaseWindModel):
    \"\"\"
    {Description}
    
    Architecture:
        {Architecture details}
    \"\"\"

    def __init__(self, config: Dict[str, Any]):
        config.setdefault("name", "{ModelName}")
        # Set default hyperparameters
        config.setdefault("hidden_size", 64)
        super().__init__(config)

        # Build layers here
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            (batch_size, output_size)
        \"\"\"
        ...
        return output
```

IMPORTANT: Generate ONLY the Python code. No markdown, no explanations outside docstrings.
The model must actually work for wind pressure Cp time series data."""


class CodeGenerator:
    """Generate PyTorch model code using Claude API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.generated_dir = Path(__file__).parent.parent / "generated_models"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    def generate_model_code(self, model_spec: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate PyTorch model code from a model specification.

        Args:
            model_spec: Dict with 'name', 'category', 'architecture_details', etc.

        Returns:
            Tuple of (code_string, file_path)
        """
        name = model_spec.get("name", "UnknownModel")
        category = model_spec.get("category", "")
        details = model_spec.get("architecture_details", {})
        innovation = model_spec.get("key_innovation", "")
        advantages = model_spec.get("advantages", [])

        prompt = f"""Generate a PyTorch model for wind pressure Cp time series forecasting.

**Model Name**: {name}
**Category**: {category}
**Architecture Details**: {details}
**Key Innovation**: {innovation}
**Advantages**: {advantages}

**Data Context**:
- Input: wind pressure coefficient (Cp) time series
- Shape: (batch_size, seq_length=100, input_size=1) — univariate by default
- Output: (batch_size, 1) — next timestep prediction
- Data range: normalized [-1, 1]
- The model should capture complex temporal patterns in turbulent wind pressure data

**Requirements**:
- Must inherit from BaseWindModel
- Must handle variable sequence lengths
- Should include proper weight initialization
- Use dropout for regularization
- Keep parameter count reasonable (<5M parameters)
- Optimize for time series with high-frequency fluctuations (1000Hz sampled)

Generate the complete Python file with the model class."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=CODE_GEN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            code = response.content[0].text

            # Clean up: remove markdown fences if present
            if "```python" in code:
                code = code.split("```python", 1)[1]
                if "```" in code:
                    code = code.rsplit("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1]
                if "```" in code:
                    code = code.rsplit("```", 1)[0]

            code = code.strip()

            # Save to file
            safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
            file_path = self.generated_dir / f"{safe_name}.py"
            file_path.write_text(code, encoding="utf-8")

            logger.info(f"Code generated for {name}: {file_path}")
            return code, str(file_path)

        except Exception as e:
            logger.error(f"Error generating code for {name}: {e}")
            raise

    def validate_code(
        self,
        code: str,
        file_path: str,
        batch_size: int = 4,
        seq_length: int = 100,
        input_size: int = 1,
    ) -> Tuple[bool, str]:
        """
        Validate that generated code compiles and produces correct output.

        Args:
            code: Python source code string
            file_path: Path where code is saved
            batch_size: Test batch size
            seq_length: Test sequence length
            input_size: Test input dimensions

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location("generated_model", file_path)
            module = importlib.util.module_from_spec(spec)

            # Ensure template is importable
            templates_dir = str(Path(__file__).parent.parent / "templates")
            if templates_dir not in sys.path:
                sys.path.insert(0, templates_dir)
            parent_dir = str(Path(__file__).parent.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            spec.loader.exec_module(module)

            # Find the model class (first class that inherits from BaseWindModel)
            from agents.modeler.templates.model_template import BaseWindModel
            model_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and issubclass(attr, BaseWindModel)
                        and attr is not BaseWindModel):
                    model_class = attr
                    break

            if model_class is None:
                return False, "No class inheriting from BaseWindModel found in generated code"

            # Instantiate with default config
            config = {
                "input_size": input_size,
                "output_size": 1,
                "seq_length": seq_length,
            }
            model = model_class(config)

            # Test forward pass
            x = torch.randn(batch_size, seq_length, input_size)
            output = model(x)

            # Validate output shape
            expected_shape = (batch_size, 1)
            if output.shape != expected_shape:
                return False, f"Output shape {output.shape} != expected {expected_shape}"

            # Check parameters
            param_count = model.count_parameters()
            if param_count == 0:
                return False, "Model has 0 trainable parameters"
            if param_count > 10_000_000:
                return False, f"Model has too many parameters: {param_count:,} (limit: 10M)"

            logger.info(f"Validation successful: {model_class.__name__} ({param_count:,} params)")
            return True, f"OK — {model_class.__name__}, {param_count:,} parameters, output shape {output.shape}"

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Validation failed: {e}")
            return False, f"Validation error: {e}\n{tb}"

    def fix_code(self, code: str, error_message: str, model_spec: Dict[str, Any]) -> str:
        """
        Ask Claude to fix broken generated code.

        Args:
            code: Original code that failed
            error_message: Error from validation
            model_spec: Original model specification

        Returns:
            Fixed code string
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=CODE_GEN_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"I tried generating a model for {model_spec.get('name', '')} but got an error."},
                    {"role": "assistant", "content": f"Here was my previous attempt:\n```python\n{code}\n```"},
                    {"role": "user", "content": f"This code produced an error:\n{error_message}\n\nPlease fix the code. Return ONLY the corrected Python code."},
                ],
            )

            fixed_code = response.content[0].text
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python", 1)[1].rsplit("```", 1)[0]
            return fixed_code.strip()

        except Exception as e:
            logger.error(f"Code fix failed: {e}")
            return code  # Return original if fix fails
