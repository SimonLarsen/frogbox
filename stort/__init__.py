"""
.. include:: ../README.md
"""
__version__ = "0.2.3"

from .pipelines.supervised import SupervisedPipeline  # noqa: F401
from .config import read_json_config  # noqa: F401
from .utils import load_model_checkpoint  # noqa: F401
from ignite.engine import Events  # noqa: F401
