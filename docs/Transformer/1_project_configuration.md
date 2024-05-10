# Project configuration

## Execution environments

Depending on the execution environment (local workspace w/o GPU, tests, Kube cluster...), the data processing pipeline and the model architecture need to be adjusted. The environment must be specified prior to running the code, in the environment variable `TRANSFORMER_ENV`. This variable is mainly used in `transformer/utils/config.py`, in order to determine which config files to use. The environment can be changed later at runtime, see [Training pipeline](6_training_pipeline.md) for more details.

⚠️ Another part of the project configuration is defined in the `charts` folder. These settings are used by the *service infra* to build the execution environment.

### Configuration files

The code configuration is defined in YAML files in `transformer/config`. Each subfolder contains the configuration for a given execution environment. Configurations are written to supersede each other, with the base settings defined in the `base/` folder. The superseding order is defined in `transformer/utils/config.py`.

The environments are:
- base: not an environment in itself, but provides a default value for all parameters
- local: personal workspace/laptop environment
- local-gpu: workspace gpu-01, used to make full model training
- test: environment used when running pytest
- lightflow: environment used when training the model in the Lightflow cluster
- optuna: same as lightflow, but for hyperparameter optimization

⚠️ The environment variable `TRANSFORMER_ENV` must not be confused with the Poetry dependency groups, defined in `pyproject.toml`.

### Accessing the configuration in the code

Example :

```python
from transformer.utils.config import get_config

cfg, paths = get_config()

# print the last day of the train set
print(cfg.data.max_train)
# print whether the Laplacian embedding is used as a model input 
print(cfg.model.use_inputs.laplacian_embedding)
```

ℹ️ The configuration is only loaded once and then stored in the `ConfigContext` class, so calling `get_config()` again is free.

### Modifying the configuration

The configuration can be modified after initialization through two facilities:
- `set_config(env)` can be used to switch from one environment to another.
- `set_config_attribute(cfg, key, value)` allows modifying an attribute of the configuration in-place.

## File paths

The names of the files required by the code (training data, pre-computed statistics or embeddings, ...) are encoded in the file `transformer/config/paths.yaml`. This allows changing the names of the files and their location without having to replace things in the code itself. The `paths.yaml` files contains two types of entries, depending on whether their name ends with `_f`:

- For regular entries (e.g. `optuna_db`), their value is the name of the related file. It can simple be accessed with `paths.optuna_db`
- Entries whose name ends with `_f` indicate that the path is a Python f-string, i.e., it contains some empty spaces `{}` that need to be filled. For instance
  - If we define the snapshots location `snapshot_f: ${snapshots_per_version}/{}/{}.pickle.zst`,
  - Then, with `paths.snapshot_f.format("2023-11-17", "2024-03-06T12:34:00+00:00")`, we would obtain the snapshot at time 12:34:00 of day 2023-11-17.

### Path resolution

Some entries in `paths.yaml` may depend on other paths, or on other code settings. Two replacements are done automatically:
- Substrings of the form `${some_path}` are replaced by the value of `some_path` in `paths.yaml` (this is done recursively in the right order)
- Substrings of the form `${config:some.code.setting}` are replaced by the value of the corresponding code setting: for instance, `models/model_with_${config:model.depth}_layers.pickle` would become `model_with_3_layers.pickle` if `cfg.model.depth = 3`.

The replacements are done with [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/).

### Accessing the paths in the code

As for the code settings, the `paths` variable is obtained with `transformer.utils.config.load_config()`.