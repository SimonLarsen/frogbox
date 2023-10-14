"""
.. include:: ../README.md
"""
__version__ = "0.1.4"

from .pipelines.supervised import train_supervised  # noqa: F401
from .config import read_json_config  # noqa: F401
from .utils import load_model_checkpoint  # noqa: F401
from ignite.engine import Events  # noqa: F401
