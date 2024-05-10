import hashlib
import json
import os
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

# e.g., when loading environment "local", we will first load the  variables in env "base", then override with "local"
# config_load_order specifies the loading order
config_load_order = {
    "base": ["base"],  # only used in the check steps of the CI pipeline
    "test": ["base", "test"],  # used in pytest environment
    "local": ["base", "local"],  # used on the machine where the code is developed
    "local-gpu": ["base", "local-gpu"],  # used on a local machine with a GPU
    "lightflow": ["base", "local-gpu", "lightflow"],  # execution in lightflow cluster
    "optuna": ["base", "local-gpu", "lightflow", "optuna"],  # hyperparameter optimization
}


class Config:
    """
    Wrapper class around an Omegaconf DictConfig.
    This wrapper allows changing the configuration behind the scenes.
    """

    _config: DictConfig

    def __init__(self, _config: DictConfig) -> None:
        self._config = _config

    def __getattr__(self, __name: str) -> Any:
        if __name == "_config":
            return super().__getattribute__(__name)
        else:
            return self._config.__getattr__(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "_config":
            super().__setattr__(__name, __value)
        else:
            self._config.__setattr__(__name, __value)

    def set_dictconfig(self, _config: DictConfig) -> None:
        self._config = _config

    @property
    def dictconfig(self) -> DictConfig:
        return self._config


class ConfigContext:
    config: Config | None = None
    paths: Config | None = None


def get_config() -> tuple[Config, Config]:
    """
    Returns: a tuple (cfg, paths) of OmegaConf configs.
    - cfg: the configuration of the code
    - paths: the path to each file used by the code (some paths are f-strings)
    """

    if (ConfigContext.config is not None) and (ConfigContext.paths is not None):
        return ConfigContext.config, ConfigContext.paths
    else:
        env = os.getenv("TRANSFORMER_ENV", "")
        set_config(env)
        return get_config()


def set_config(env: str) -> None:
    """
    Loads and enrich the YAML config files.
    Enrich = adds dynamically computed variables.

    The env variable (local, ..., see config_load_order.keys()) is retrieved from the
    environment variable TRANSFORMER_ENV.
    """

    assert (
        env in config_load_order.keys()
    ), f"Invalid environment provided (must be in {list(config_load_order.keys())}), got {env}."

    cfg = OmegaConf.create()

    cfg_files = list(OmegaConf.load("transformer/config/config_base.yaml"))  # type: list[str]
    for cfg_name in cfg_files:
        cfg.__setattr__(cfg_name, OmegaConf.create())

    for e in config_load_order[env]:
        cfg_folder = f"transformer/config/{e}"
        for cfg_filename in cfg_files:
            path = os.path.join(cfg_folder, f"{cfg_filename}.yaml")
            if os.path.exists(path):
                cfg_file = OmegaConf.load(path)
                cfg.__setattr__(cfg_filename, OmegaConf.merge(cfg.__getattr__(cfg_filename), cfg_file))

    cfg.env = env

    paths = compute_paths(cfg, "transformer/config/paths.yaml")

    if ConfigContext.config is None:
        ConfigContext.config = Config(cfg)
    else:
        ConfigContext.config.set_dictconfig(cfg)

    if ConfigContext.paths is None:
        ConfigContext.paths = Config(paths)
    else:
        ConfigContext.paths.set_dictconfig(paths)


def compute_paths(cfg: DictConfig, paths_file: str) -> DictConfig:
    """
    Given the loaded OmegaConf config, loads and fills the template file paths
    """

    paths = OmegaConf.load(paths_file)
    cfg_dict = config_to_dict(cfg)

    # In the paths, replace ${config:param} by the corresponding config value
    # See https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation
    OmegaConf.clear_resolver("config")
    OmegaConf.register_new_resolver("config", lambda param: cfg_dict[param])
    OmegaConf.resolve(paths)

    return DictConfig(paths)


def config_to_dict(cfg: DictConfig | Config) -> dict[str, str | int | float | bool]:
    if isinstance(cfg, Config):
        cfg = cfg._config
    cfg_dict = OmegaConf.to_object(cfg)  # Turn config to nested dictionaries
    return _linearize_dict(cfg_dict)  # type: ignore


def hash_config(cfg: DictConfig | Config) -> str:
    """
    Hash an OmegaConf config into a string. Used to create unique file names associated to a given state of the config.
    """
    if isinstance(cfg, Config):
        cfg = cfg._config
    cfg_dict = config_to_dict(cfg)
    cfg_str = json.dumps(cfg_dict, sort_keys=True)
    cfg_hash = hashlib.sha512(str(cfg_str).encode("utf-8")).hexdigest()
    return cfg_hash


def _linearize_dict(
    input_dict: dict[str, Any], running_prefix: str = "", result_dict: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Example usage:
    a = {'c': 1, 'd': {'e': 2, 'f': 3}}
    _linearize_dict(a)
    >>> {'c': 1, 'd.e': 2, 'd.f': 3}
    """

    if result_dict is None:
        result_dict = {}

    for key, value in input_dict.items():
        if running_prefix == "":
            result_key = key
        else:
            result_key = f"{running_prefix}.{key}"

        if isinstance(value, dict):
            _linearize_dict(value, result_key, result_dict)
        else:
            result_dict[result_key] = value

    return result_dict


def set_config_attribute(cfg: DictConfig | Config, key: str, value: Any) -> None:
    """
    Modify an attribute in cfg.
    For instance, if key="training.batch_size" and value=64, the function will set cfg.training.batch_size = value
    """
    if isinstance(cfg, Config):
        set_config_attribute(cfg._config, key, value)
    else:
        key_chain = key.split(".")
        for k in key_chain[:-1]:
            cfg = cfg.__getattr__(k)
        cfg.__setattr__(key_chain[-1], value)


if __name__ == "__main__":
    os.environ["TRANSFORMER_ENV"] = "local"
    cfg, paths = get_config()
    print(paths.config_statistics_f, "\n")
    print(OmegaConf.to_yaml(cfg.model))
    print(cfg.data.stats.use_local_diff_delay)
