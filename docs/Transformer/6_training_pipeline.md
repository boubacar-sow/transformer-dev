# Training pipeline

## DAG structure

The training procedure is written as a [Lightflow](https://gitlab.com/osrdata/infra/lightflow/) DAG. It can be launched locally by running the commands:
```sh
mlflow server # must be running in the background, in a dedicated folder
export TRANSFORMER_ENV=local # or any other environment, see Project Configuration
python -m lightflow.runners.local_runner model_training
```

The DAG executes several steps (see `lightflow_dags/model_training.py`):
- `fetch_files` downloads the training data, embeddings, statistics required for training
- `compute_statistics_and_embeddings` computes... well... the statistics and... the embeddings
- `train_model` contains the actual training loop (see below)
- `evaluate_model` contains the test step: it computes a handful of metrics and plots.

## DAG arguments

The DAG can be provided with three arguments:
- the environment `env` (see [Project configuration](1_project_configuration.md))
- an existing MLFlow run ID, `warm_start_run_id`, to use the pre-trained weights of the given run
- if `warm_start_run_id` is defined, the additional boolean argument `resume_run` determines whether or not a new MLFlow run is created
These arguments are provided with the `--configuration` argument of the lightflow CLI, as here:
```sh
python -m lightflow.runners.local_runner dag_name --configuration '{"env": "local-gpu", "run_id": "fzou768khziu78", "warm_start_run_id": "uhoyigiy766"}'
```
The three arguments are optional. Additionally, if not found in the CLI arguments, default values can be provided through environment variables:
- `TRANSFORMER_ENV` for `env`
- `TRANSFORMER_WARM_START_RUN_ID` for `warm_start_run_id`
- `TRANSFORMER_RESUME_RUN` for `resume_run`, which must be set `0` or `1`

⚠️ The code currently requires either the `TRANSFORMER_ENV` to be defined, even though it may be overriden by the DAG configuration later.

## Training loop

The training is done with [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html), a package that automates the training and provides lot of convenient parameters to configure the training behavior. We mainly use three objects from Pytorch Lightning:
- the `LightningModule` class, which is a wrapper class around `torch.nn.Module`, and provides a function for each step of the training pipeline. By default, all these functions are empty, and writing the pipeline amounts to filling the functions corresponding to the pipeline steps where we want to to things.
- the `LightningDataModule` class, which is a light class that generates Pytorch DataLoaders for the train, validation, test and prediction datasets. Again, by default it provides empty functions, and we fill them with the desired behavior.
- the `Trainer` class, that actually performs the training and testing steps.

Throughout the training and the testing steps, metrics, hyperparameters, figures and checkpoints are logged to [MLFlow](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html). If the code is running locally, an MLFlow server needs to be running on the machine. It can simply be launched by running `mlflow server` in the folder where we want MLFlow to store its data. To visualize the logged data, the MLFlow interface can be accessed by default at `http://127.0.0.1:5000`, but this address can be configured in `cfg.training.mlflow.tracking_uri`.

ℹ️ The logging to MLFlow can be turned off by setting `cfg.training.mlflow.use_mlflow` to `false` in the configuration file of your environment (e.g., in `transformer/config/local/training.yaml` if you typically start with the `local` environment).

## Hyperparameter tuning DAG

The code provides a DAG to optimize the hyperparameters using [Optuna](https://optuna.org/). It can be launched with:

```sh
python -m lightflow.runners.local_runner hyperparameter_tuning
```

The script produces a series of nested MLFlow runs, as well as experiment details recorded by Optuna in a SQLite database stored in `paths.optuna_db` in S3. If `cfg.data.download_data_from_s3`, the DB will be download from S3 (if it exists). If `cfg.data.save_results_to_s3` is True, the (updated) DB will be saved to S3 at the end of the DAG.

⚠️ When running the DAG in the Lightflow cluster, the environment needs to be specified with the `--configuration` argument to override the `TRANSFORMER_ENV` variable, which cannot be changed for each DAG in the Lightflow cluster (see [Lightflow doc](https://docs.dev.dgexsol.fr/lightflow/dags/5_running_DAGs.html) and [utnm doc](https://docs.dev.dgexsol.fr/utnm/utnm/lightflow.html)). The command would look like this:
```sh
# Run the DAG locally
export TRANSFORMER_ENV=optuna
python -m lightflow.runners.local_runner hyperparameter_tuning
# Run the DAG in the Lightflow cluster with utnm
utnm lightflow run lightflow://dev/transformer/hyperparameter_tuning --configuration '{"env": "optuna"}'
```

## Evaluation DAG

The DAG `model_evaluation` allows to 