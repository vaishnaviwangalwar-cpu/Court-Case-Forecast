"""MLflow logging when installed; JSON-lines fallback otherwise."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src import config


def log_run(name: str, params: dict, metrics: dict, out_dir: Path = config.ARTIFACT_DIR) -> None:
    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{config.MLRUNS_DIR}")
        mlflow.set_experiment("court-case-forecast")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in flat.items() if v == v})
    except Exception:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "runs.jsonl", "a") as f:
            f.write(json.dumps({"ts": time.time(), "name": name, "params": params, "metrics": flat}, default=str) + "\n")
