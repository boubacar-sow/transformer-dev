from lightflow.dags.dag import Dag  # type: ignore
from lightflow.tasks.task import task  # type: ignore

from transformer.training.hyperparameter_tuning import tune_transformer
from transformer.utils import dag_steps
from transformer.utils.logger import logger
from transformer.utils.mlflow import upload_logs_to_mlflow

with Dag(name="hyperparameter_tuning"):

    @task()  # type: ignore
    def init() -> str:
        dag_arguments = dag_steps.DagArgs(env="optuna", warm_start_run_id=None, resume_run=False)
        _, _, mlflow_experiment_id = dag_steps.init(dag_arguments)
        return mlflow_experiment_id

    @task()  # type: ignore
    def fetch_files() -> None:
        logger.info("===Downloading and preprocessing training related files===")
        dag_steps.fetch_files()

    @task()  # type: ignore
    def tune_model(experiment_id: str) -> None:
        tune_transformer(experiment_id)

    @task()  # type: ignore
    def upload_logs() -> None:
        upload_logs_to_mlflow()

    mlflow_experiment_id = init()
    fetch_files = fetch_files()
    tune_model = tune_model(mlflow_experiment_id)
    upload_logs = upload_logs()
    mlflow_experiment_id >> fetch_files >> tune_model >> upload_logs
