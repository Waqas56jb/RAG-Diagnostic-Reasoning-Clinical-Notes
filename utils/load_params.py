# utils/load_params.py
import yaml
from pathlib import Path

def load_params():
    """
    Load and return parameters from params.yaml at project root.
    """
    params_path = Path(__file__).resolve().parent.parent / "params.yaml"
    params_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent dirs exist
    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found at {params_path}")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)