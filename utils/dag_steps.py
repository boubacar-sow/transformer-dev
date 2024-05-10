import os
from dataclasses import dataclass
from functools import lru_cache

import torch
from lightflow.dags.dag_run_context import DagRunContext  # type: ignore

from transformer.inputs.resources.pr_download import download_and_process_pr_data
from transformer.inputs.resources.pr_graph import build_pr_graph
from transformer.utils.config import get_config, set_config
from transformer.utils.file_system import FileSystem
from transformer.utils.logger import LOGGING_LEVELS, init_logging, logger
from transformer.utils.misc import set_random_seeds
from transformer.utils.mlflow import init_mlflow
from transformer.utils.s3 import download_training_data_from_s3

cfg, paths = get_config()


@dataclass
class DagArgs:
    env: str
    warm_start_run_id: str | None
    resume_run: bool | None


@lru_cache()
def get_dag_arguments() -> DagArgs:
    dag_configuration = DagRunContext.get_current_context().dag_configuration
    env = dag_configuration.get("env", os.getenv("TRANSFORMER_ENV"))
    warm_start_run_id = dag_configuration.get("warm_start_run_id", os.getenv("TRANSFORMER_WARM_START_RUN_ID"))
    resume_run_env = os.getenv("TRANSFORMER_RESUME_RUN")
    if resume_run_env is not None:
        assert resume_run_env in ["0", "1"], "TRANSFORMER_RESUME_RUN must be either 0 or 1."
        resume_run = bool(int(resume_run_env))
    else:
        resume_run = None
    resume_run = dag_configuration.get("resume_run", resume_run)
    return DagArgs(env=env, warm_start_run_id=warm_start_run_id, resume_run=resume_run)


def init(dag_arguments: DagArgs) -> tuple[str, str | None, str]:
    """
    Change the configuration environment. Initialize logging, mlflow runs and random seeds.
    Returns:
    - the current mlflow run ID
    - the mlflow run ID used for warm start
    - the experiment ID
    NB: If mlflow is not used, the mlflow run ID and the experiment ID will both be "".
        If no warm start argument is given, warm_start_run_id will be "".
    """

    # Get the configuration environment. Default value, use the env variable TRANSFORMER_ENV instead
    # Set the config environment
    set_config(dag_arguments.env)

    # Set logging parameters
    init_logging(log_file=paths.logs_file, logging_level=LOGGING_LEVELS[cfg.training.logging])

    # Set global random seed
    set_random_seeds(cfg.training.random_seed)

    warm_start_run_id = dag_arguments.warm_start_run_id

    # Start mlflow run (only if cfg.training.mlflow.use_mlflow is True)
    if cfg.training.mlflow.use_mlflow:
        run_id = warm_start_run_id if (warm_start_run_id is not None and dag_arguments.resume_run) else None
        run_id, experiment_id = init_mlflow(run_id=run_id)
    else:
        run_id, experiment_id = "", ""

    logger.info(f"DAG configuration: {str(dag_arguments)}")
    logger.info(f"MLFlow run ID: {run_id}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")

    return run_id, warm_start_run_id, experiment_id


def fetch_files() -> None:
    """
    Download or sync the required training data.
    """
    # Request remote files
    if cfg.data.download_data_from_s3:
        logger.info("Retrieving training data from S3.")
        download_training_data_from_s3()
        pr_downloaded = download_and_process_pr_data()
    else:
        logger.info("Using locally stored data.")
        pr_downloaded = False

    fs = FileSystem()
    if pr_downloaded or not fs.exists(paths.PRs_graph_networkx):
        logger.info("Computing PRs graph.")
        build_pr_graph()
