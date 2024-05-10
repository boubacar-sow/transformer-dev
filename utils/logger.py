import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("transformer")

LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def init_logging(log_file: str, logging_level: int = logging.INFO) -> None:
    """
    Configures the "transformer" Python logger. The logs are both displayed in console and saved to the given file.

    Args:
    log_file: path to logs file.
    logging_level: if provided, can be logging.DEBUG/INFO/WARNING/ERROR.
    """

    path = Path(log_file)
    os.makedirs(path.parent, exist_ok=True)

    _formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _handler_err = logging.StreamHandler(sys.stderr)
    _handler_err.setFormatter(_formatter)
    _handler_file = logging.FileHandler(log_file, mode="w")
    _handler_file.setFormatter(_formatter)

    logger = logging.getLogger("transformer")
    logger.setLevel(logging_level)
    logger.handlers.clear()
    logger.addHandler(_handler_err)
    logger.addHandler(_handler_file)
    logger.propagate = False
