# Transformer training repository


This repo hosts the Transformer project, a ML model to predict train delays. It contains the code to train the neural network and perform real-time predictions, as well as data preprocessing and output postprocessing functions.

More detailed information and explanations can be found:
- ⌨️ on the code structure in the [documentation](https://docs.dev.dgexsol.fr/transformer/Transformer/Training%20pipeline.html)
- 📖 on the model and the inputs processing on [Confluence](https://pdi-dgexsolutions-sncf.atlassian.net/wiki/spaces/RE/pages/192741442/Transformer).

## 🛠️ Getting  on Workspaces

First, create the poetry environment and make it use a suitable version of Python:
```sh
pyenv install 3.11.4
pyenv global 3.11.4
poetry env use /home/9509543k/.pyenv/shims/python
```

Next, install the poetry environment:
```sh
poetry install --with dev
```

Finally, install additional dependencies currently not specified in `pyproject.toml`:
```sh
# Only for machines with a GPU
pip install torch==2.1.1
# Only for machines with no GPU
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
# Pytorch Lightning and MLFlow
pip install lightning==2.1.2 mlflow==2.11.3
```

ℹ️ Pytorch and the libraries that depend on it are installed separately because poetry is currently not able to handle multiple versions of the same package, even for distinct dependency groups (see [this issue](https://github.com/python-poetry/poetry/issues/6409#issuecomment-1310173723) and the comments below).

## ⚙️ Environment variables and project configuration

The code uses the following environment variables:
- `TRANSFORMER_ENV`: determines which settings are used to run the code (the settings are stored in `f"transformer/config/{TRANSFORMER_ENV}"`, more information [here](https://docs.dev.dgexsol.fr/transformer/Transformer/Project%20configuration.html))
- `TRANSFORMER_WARM_START_RUN_ID` (optional): if set, the training will start from the checkpoint of the given MLFlow run ID

Running the code also requires additional environment variables and permissions:
- The Cerbère tokens in `chart/values.yaml` must be added as [personal tokens](https://pdi-dgexsolutions-sncf.atlassian.net/wiki/spaces/INF/pages/127238145/Token) by the developer
  - `mlflow_transformer` is mandatory (cf [the docs](https://docs.dev.dgexsol.fr/mlflow/))
  - `pr_data_transformer` is mandatory if no pre-processed data is available
  - `lightflow_trigger_transformer` is used when launching DAGs in the Lightflow cluster
- The developer needs the AWS role `pdi-re-transformer-rw-augias-ro`

## 💾 Running the code

### Locally

```sh
poetry shell
mlflow server # running in the background, in a dedicated folder
export TRANSFORMER_ENV=local # set the environment to local
python -m lightflow.runners.local_runner model_training
```

The necessary training data (snapshots and additional files) must be downloaded and stored in the folder given in `config.data.local_folder` / `config.data.bucket_name` (see `transformer/config/base/data.yaml` settings).

### In the Lightflow cluster
Using [utnm](https://gitlab.com/osrdata/libraries/utnm):
```
utnm lightflow run lightflow://dev/transformer/model_training
```


## 🗺️ Code organization

All the code and configuration are stored in `transformer`, with other folders dedicated to building (`dockerfiles`), deploying (`chart`) and executing (`lightflow_dag`) the code. The main code is divided into three parts (more detailed structure [here](https://docs.dev.dgexsol.fr/transformer/Transformer/Code%20structure.html)):
- `inputs`: functions to load the input data and compute the embeddings of trains and/or PRs,
- `training`: model architecture and training pipeline with Pytorch Lightning,
- `outputs`: metrics definition, and currently unused prediction postprocessing scripts.

Additionally:
- `config` contains the YAML configuration of the code for various environments,
- `tests` contains ... the tests,
- `utils` contains various functions used throughout the code, e.g. YAML configuration loading.

