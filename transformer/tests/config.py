from transformer.utils.config import get_config, set_config_attribute

cfg, paths = get_config()


def test_set_config() -> None:
    cfg.training.batch_size = 1
    set_config_attribute(cfg, "training.batch_size", 2)
    assert cfg.training.batch_size == 2
