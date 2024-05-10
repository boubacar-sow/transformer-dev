import lightning.pytorch as pl
from lightflow.dags.dag import Dag  # type: ignore
from lightflow.tasks.task import task  # type: ignore

from transformer.inputs.data_module import SnapshotDataModule
from transformer.inputs.embeddings.embeddings import compute_embeddings
from transformer.inputs.snapshots.statistics import compute_statistics
from transformer.training.modules import TransformerLightningModule
from transformer.training.test_model import test_model
from transformer.training.train_model import train_model
from transformer.utils import dag_steps
from transformer.utils.config import get_config
from transformer.utils.logger import logger
from transformer.utils.mlflow import upload_logs_to_mlflow

cfg, paths = get_config()

with Dag(name="model_training"):

    @task()  # type: ignore
    def init() -> tuple[str, str | None]:
        dag_arguments = dag_steps.get_dag_arguments()
        run_id, warm_start_run_id, _ = dag_steps.init(dag_arguments)
        return run_id, warm_start_run_id

    @task()  # type: ignore
    def fetch_files() -> None:
        logger.info("===Downloading and preprocessing training related files===")
        dag_steps.fetch_files()

    @task()  # type: ignore
    def compute_statistics_and_embeddings_task() -> None:
        logger.info("===Computing statistics===")
        compute_statistics()
        logger.info("===Computing embeddings===")
        compute_embeddings()

    @task()  # type: ignore
    def train_step(
        run_id: str, warm_start_run_id: str | None
    ) -> tuple[TransformerLightningModule, SnapshotDataModule, pl.Trainer]:
        lightning_module, data_module, trainer = train_model(run_id, warm_start_run_id)
        return lightning_module, data_module, trainer

    @task()  # type: ignore
    def test_step(model: TransformerLightningModule, data_module: SnapshotDataModule, trainer: pl.Trainer) -> None:
        if cfg.training.test_step.skip:
            logger.info("Skipping test step.")
        else:
            test_model(model, data_module, trainer)

    @task()  # type: ignore
    def upload_logs() -> None:
        upload_logs_to_mlflow()

    run_id, warm_start_run_id = init()
    fetch_files = fetch_files()
    compute_statistics_and_embeddings_task = compute_statistics_and_embeddings_task()
    lightning_module, data_module, trainer = train_step(run_id, warm_start_run_id)
    test_step = test_step(lightning_module, data_module, trainer)
    upload_logs = upload_logs()
    run_id >> fetch_files >> compute_statistics_and_embeddings_task >> lightning_module
    test_step >> upload_logs
