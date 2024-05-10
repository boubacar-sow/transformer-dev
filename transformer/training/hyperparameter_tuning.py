from pathlib import Path

import mlflow  # type: ignore
import optuna
from omegaconf import DictConfig

from transformer.inputs.embeddings.embeddings import compute_embeddings
from transformer.inputs.snapshots.statistics import compute_statistics

# from transformer.training.losses import LOSSES
from transformer.training.train_model import train_model
from transformer.utils.config import get_config, set_config_attribute
from transformer.utils.logger import logger
from transformer.utils.s3 import copy_folder_to_s3

cfg, paths = get_config()

BOOLEAN_OPTIONS = [True, False]

# key: list_of_choices
CATEGORICAL_HYPERPARAMETERS = {
    # "data.preprocessing.transform_delay_sqrt": BOOLEAN_OPTIONS,
    # "data.preprocessing.transform_times_sqrt": BOOLEAN_OPTIONS,
    # "data.preprocessing.normalize_late": BOOLEAN_OPTIONS,
    # "data.preprocessing.normalize_times": BOOLEAN_OPTIONS,
    # "data.preprocessing.normalize_all_inputs": BOOLEAN_OPTIONS,
    # "data.preprocessing.normalize_non_eigen_inputs": BOOLEAN_OPTIONS,
    # "embeddings.laplacian_embedding.version": ["TRANSITION", "GENERAL_EIGENVALUE", "CLASSIC_RESCALED", "CLASSIC"],
    "model.use_inputs.train_type": BOOLEAN_OPTIONS,
    "model.use_inputs.nbr_train_now": BOOLEAN_OPTIONS,
    "model.use_inputs.year_day": BOOLEAN_OPTIONS,
    "model.use_inputs.week_day": BOOLEAN_OPTIONS,
    "model.use_inputs.current_time": BOOLEAN_OPTIONS,
    # "model.use_inputs.train_nextdesserte_embedding": BOOLEAN_OPTIONS,
    # "model.use_inputs.pr_nextdesserte_embedding": BOOLEAN_OPTIONS,
    # "model.use_inputs.lignepk_embedding": BOOLEAN_OPTIONS,
    "model.use_inputs.laplacian_embedding": BOOLEAN_OPTIONS,
    "model.use_inputs.node2vec_embedding": BOOLEAN_OPTIONS,
    # "model.use_inputs.geographic_embedding": BOOLEAN_OPTIONS,
    # "model.use_inputs.random_embedding": BOOLEAN_OPTIONS,
    # "training.loss": list(LOSSES.keys()),
    # "training.batch_size": [4, 8, 16, 32, 64],
}  # type: dict[str, list[int] | list[bool] | list[str]]

# key: (lower_bound, upper_bound)
INTEGER_HYPERPARAMETERS = {
    # "embeddings.laplacian_embedding.dim": (5, 40),
    # "embeddings.node2vec_embedding.dim": (5, 40),
    # "embeddings.node2vec_embedding.num_walks": (5, 40),
    # "embeddings.node2vec_embedding.walk_length": (50, 200),
    "model.depth": [1, 5],
    "model.nhead": [1, 5],
    "model.model_dim": [512, 1024],
}

# key: (lower_bound, upper_bound, use_log_scale_for_sampling)
FLOAT_HYPERPARAMETERS = {
    # "embeddings.node2vec_embedding.p": (0.5, 2, False),
    # "embeddings.node2vec_embedding.q": (0.5, 2, False),
    "training.learning_rate": (1e-6, 1e-3, True),
    "training.dropout": (0.01, 0.5, True),
}  # type: dict[str, tuple[float, float, bool]]


def suggest_config(trial: optuna.Trial) -> DictConfig:
    trial_config = cfg.dictconfig.copy()

    for key, choices in CATEGORICAL_HYPERPARAMETERS.items():
        cat_value = trial.suggest_categorical(key, choices)
        set_config_attribute(trial_config, key, cat_value)

    for key, (low, high) in INTEGER_HYPERPARAMETERS.items():
        int_value = trial.suggest_int(key, low=low, high=high)
        set_config_attribute(trial_config, key, int_value)

    for key, (low, high, log) in FLOAT_HYPERPARAMETERS.items():  # type: ignore[assignment]
        float_value = trial.suggest_float(key, low=low, high=high, log=log)
        set_config_attribute(trial_config, key, float_value)

    trial_config.model.model_dim = trial_config.model.nhead * (trial_config.model.model_dim // trial_config.model.nhead)

    return trial_config


def objective_transformer(trial: optuna.Trial, experiment_id: str) -> float:
    with mlflow.start_run(nested=True, experiment_id=experiment_id, run_name=f"optuna_{trial.number + 1}") as run:
        # Suggest a new configuration
        trial_config = suggest_config(trial)
        cfg.set_dictconfig(trial_config)

        # Compute new statistics and embeddings, if needed
        logger.info("===Computing statistics===")
        compute_statistics()
        logger.info("===Computing embeddings===")
        compute_embeddings()

        # Train the model
        trainer = train_model(run_id=run.info.run_id, trial=trial)[2]

        return trainer.logged_metrics["val_loss"]


def tune_transformer(experiment_id: str) -> None:
    logger.info("===Creating Optuna study===")
    Path(paths.optuna_storage).mkdir(parents=True, exist_ok=True)  # create path for optuna logs
    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler(seed=cfg.training.random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, storage=paths.optuna_db)

    logger.info("===Starting hyperparameter tuning===")
    study.optimize(
        lambda trial: objective_transformer(trial, experiment_id),
        n_trials=cfg.training.optuna.n_trials,
        n_jobs=1,
    )

    if cfg.data.save_results_to_s3:
        copy_folder_to_s3(paths.optuna_storage)
