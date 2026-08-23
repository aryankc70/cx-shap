"""
Shared MLflow logging helper. Uses a local file-based tracking store
(mlruns/ directory) - no external server needed, keeps everything
inside the repo's working directory for full reproducibility.
"""

import mlflow

EXPERIMENT_NAME = "cx-shap"


def setup_experiment():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(run_name: str, params: dict, metrics: dict, artifact_paths: list[str] = None):
    """metrics: flat dict of scalar metric name -> value (nested dicts get
    flattened automatically, e.g. {"pph": {"auc": 0.99}} -> "pph_auc")."""
    setup_experiment()

    def flatten(d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                flat.update(flatten(v, prefix=f"{key}_"))
            elif isinstance(v, (int, float)):
                flat[key] = v
        return flat

    flat_metrics = flatten(metrics)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(flat_metrics)
        if artifact_paths:
            for path in artifact_paths:
                mlflow.log_artifact(path)