from lightflow.dags.dag import Dag  # type: ignore
from lightflow.tasks.task import task  # type: ignore

from transformer.training.test_model import test_model
from transformer.training.train_model import get_lightning_objects
from transformer.utils import dag_steps
from transformer.utils.config import get_config
from transformer.utils.logger import logger
from transformer.utils.mlflow import upload_logs_to_mlflow

cfg, paths = get_config()


with Dag(name="model_evaluation"):

    @task()  # type: ignore
    def init() -> str:
        dag_arguments = dag_steps.get_dag_arguments()
        dag_arguments.resume_run = True  # always stay in the same run for evaluation
        assert (
            dag_arguments.warm_start_run_id is not None
        ), "The environment variable TRANSFORMER_WARM_START_RUN_ID must not be empty"
        dag_steps.init(dag_arguments)  # log metrics in the same run than the one where the model was trained
        return dag_arguments.warm_start_run_id

    @task()  # type: ignore
    def fetch_files() -> None:
        logger.info("===Downloading and preprocessing training related files===")
        dag_steps.fetch_files()

    @task()  # type: ignore
    def evaluate_model(run_id: str) -> None:
        lightning_module, data_module, trainer = get_lightning_objects(run_id=run_id)
        test_model(lightning_module, data_module, trainer)

    @task()  # type: ignore
    def upload_logs() -> None:
        upload_logs_to_mlflow(log_directory="eval_logs")

    mlflow_run_id = init()
    fetch_files = fetch_files()
    evaluate_model = evaluate_model(mlflow_run_id)
    upload_logs = upload_logs()
    mlflow_run_id >> fetch_files >> evaluate_model >> upload_logs
