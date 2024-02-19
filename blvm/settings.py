"""Settings file for blvm package"""

import os
import logging

from typing import Any

from blvm.utils.logger import ColoredLogsFormatter


ROOT_PATH = __file__.replace("/blvm/settings.py", "")
ENV_FILE = os.path.join(ROOT_PATH, "BLVM.env")


def read_env_file():
    if not os.path.exists(ENV_FILE):
        return dict()

    with open(ENV_FILE, "r") as envfile_buffer:
        env_text = envfile_buffer.read()
    return dict([line.split("=") for line in env_text.splitlines()])


ENV = read_env_file()


def write_blvm_envvar(envvar: str, envvar_value: str):
    """Write an envvar to the ENV_FILE"""
    with open(ENV_FILE, "a") as envfile_buffer:
        envfile_buffer.write(f"{envvar}={envvar_value}\n")


def require_blvm_envvar(envvar):
    """Request user for value of a required envvar and write it to the ENV_FILE"""
    envvar_value = input(f"\nRequired environment variable {envvar} not set.\n\nPlease specify:\n> ")
    write_blvm_envvar(envvar, envvar_value)
    return envvar_value


def get_blvm_envvar(envvar, default: Any = None, reflect: bool = False):
    """Retrieve the value of an envvar.

    In prioritized order, returns the value found in `os.environ`, `ENV_FILE`, `default`.

    If `default` is `None`, and the envvar is not in `os.environ` or `ENV_FILE`, requires the envvar to be set and 
    requests it from the user at runtime.

    If `reflect` is `True`, reflects the retrieved value into `os.environ` in all cases.
    """
    if envvar in os.environ:
        return os.getenv(envvar)

    if envvar in ENV:
        value = ENV[envvar]
    elif default is None:
        value = require_blvm_envvar(envvar)
    else:
        value = default

    if reflect:
        os.environ[envvar] = value

    return value


# logging
LOG_FORMAT_DEFAULT = "%(asctime)-15s %(module)-20s : %(levelname)-8s | %(message)s"
LOG_FORMAT = get_blvm_envvar("BLVM_LOG_FORMAT", default=LOG_FORMAT_DEFAULT)
LOG_LEVEL = get_blvm_envvar("BLVM_LOG_LEVEL", default="INFO")

console = logging.StreamHandler()
formatter = ColoredLogsFormatter(format=LOG_FORMAT)
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
logging.getLogger().setLevel(logging.getLevelName(LOG_LEVEL))  # set level for root logger
logging.getLogger("wandb").setLevel(logging.WARNING)  # set level for wandb logger

# data directories
DATA_ROOT_DIRECTORY = get_blvm_envvar("BLVM_DATA_ROOT_DIRECTORY", default=None)
DATA_DIRECTORY = os.path.join(DATA_ROOT_DIRECTORY, "data")
SOURCE_DIRECTORY = os.path.join(DATA_ROOT_DIRECTORY, "source")
VOCAB_DIRECTORY = os.path.join(DATA_ROOT_DIRECTORY, "vocabularies")

# set wandb directory to inside data directory
WANDB_DIR = get_blvm_envvar("WANDB_DIR", default=DATA_ROOT_DIRECTORY, reflect=True)
CHECKPOINT_DIR = os.path.join(WANDB_DIR, "wandb")

# make directories
for path in [DATA_DIRECTORY, SOURCE_DIRECTORY, VOCAB_DIRECTORY, WANDB_DIR]:
    os.makedirs(path, exist_ok=True)
