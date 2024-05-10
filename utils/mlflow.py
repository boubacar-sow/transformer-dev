import glob
import os
from typing import Optional

import mlflow  # type: ignore

from transformer.training.modules import compute_input_dim
from transformer.utils.config import config_to_dict, get_config
from transformer.utils.logger import logger

cfg, paths = get_config()


def init_mlflow(run_id: Optional[str] = None) -> tuple[str, str]:
    """
    If run_id is provided, get the corresponding mlflow run and return its experiment ID.
    Else, start an mlflow run and return (run ID, experiment ID). The experiment name depends on TRANSFORMER_ENV.

    Args:
        run_id: mlflow run ID
    """

    mlflow.set_tracking_uri(cfg.training.mlflow.tracking_uri)

    # Get the experiment ID
    experiment_name = f"transformer_{cfg.env}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    if run_id is not None:
        run = mlflow.start_run(run_id=run_id)  # Retrieve the run
        experiment_id = run.info.experiment_id
    else:
        run = mlflow.start_run(experiment_id=experiment_id)  # Start a new run
        run_id = run.info.run_id

    return run_id, experiment_id


def load_checkpoint_from_mlflow(run_id: str) -> str:
    """
    Returns the path to the last checkpoint.
    """

    # Get the warm start run
    warm_start_run = mlflow.get_run(run_id)

    # Get its model config
    run_cfg = warm_start_run.data.params
    run_cfg = {k[6:]: v for k, v in run_cfg.items() if k.startswith("model.")}
    del run_cfg["input_dim"]  # the key input_dim is logged in MLFlow, but it is not a config parameter

    # Check if it matches the current model architecture
    current_cfg = config_to_dict(cfg.model)
    current_cfg = {k: str(v) for k, v in current_cfg.items()}

    if current_cfg != run_cfg:
        s1 = set(current_cfg.keys())
        s2 = set(run_cfg.keys())
        logger.error(
            f"Keys difference between warm start run config and local config: {str(s1 ^ s2)}"
        )  # s1 ^ s2 is the symmetric difference between sets
        raise Exception("Trying to warm start a model with different hyperparameters.")

    # Download the model checkpoints
    dst_path = os.path.join(paths.warm_start, run_id)
    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=dst_path, artifact_path="model/checkpoints")

    # Keep only the files with a checkpoint format
    files = glob.glob(f"{dst_path}/**", recursive=True)
    checkpoints = [f for f in files if f.endswith(".ckpt")]

    if len(checkpoints) == 0:
        raise Exception("Found no model artifact for the given run ID.")

    if len(checkpoints) > 1:
        logger.info("Found several model artifact files. Choosing the most advanced checkpoint.")
        checkpoints = sorted(checkpoints)

    return checkpoints[-1]


def log_config_to_mlflow() -> None:
    cfg_dict = config_to_dict(cfg)
    cfg_dict["model.input_dim"] = compute_input_dim()  # add input_dim, as it is computed at runtime
    mlflow.log_params(cfg_dict)


def upload_logs_to_mlflow(log_directory: str = "logs") -> None:
    """
    Save logs to MLFlow.
    """
    mlflow.log_artifacts(local_dir=paths.logs_dir, artifact_path=log_directory)
