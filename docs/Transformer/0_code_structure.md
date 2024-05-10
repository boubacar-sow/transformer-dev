# Code structure

## Inputs

- `data_structures.py`: the data structures used throughout the code
- `data_loading.py`: the main functions for loading the training data (i.e., preprocessed snapshots)
- `data_module.py`: wrapper classes around the data loading function (torch Dataset and Lightning DataModule)
- `snapshots/`: contains functions to:
  - `snapshot_files.py`: list snapshots
  - `statistics.py`: compute statistics
  - `time_transform.py`: use these statistics to normalize times and delays
- `embeddings/`: functions to compute and load embeddings (see [Embeddings](2_statistics.md))
- `resources/`: information on Points Remarquables (code CI/CH, ID Gaia, name, ...) and on train types (using `circulation_numerotation`)
- `utils/`: mainly contains filters and tagger classes (see [Filters and tags](4_filters_and_tags.md))

## Training

- `modules.py`: the model in Pytorch Lightning Module format. This includes the train, validation and test steps
- `train_model.py`: the main function to run the training loop using Lightning's Trainer class
- `test_model.py`: the test step of the training phase
- `hyperparameter_tuning.py`: Optuna study performing multiple trainings
- `metrics/losses.py`: miscellaneous loss functions, gathered in the `losses.LOSSES` dictionnary
- `metrics/compute_metrics.py`: computes test metrics (called during the test step) 
- `metrics/metrics.py`: miscellaneous metrics, gathered in the `metrics.METRICS` dictionnary
- `metrics/plot_metrics.py`: plot model error as a function of space (PR location) or time

## Outputs

- `post_processing/`: currently unused

## Utils

- `config.py`: see Project configuration
- `dag_steps.py`: functions called at the start of the training procedures (start mlflow run, set random seeds, configure logging, etc)
- `file_system.py`: interface to read and write files, possibly in zstandard compressed format
- `loaders.py`: utility functions to read and write files in common formats
- `misc.py`: well... misc...
- `mlflow.py`: functions to start mlflow runs and log various things
- `s3.py`: functions to download the training data from a S3 bucket
- `stats_counter.py`: see [Statistics](2_statistics.md)
- `times.py`: functions to manipulate dates and time intervals