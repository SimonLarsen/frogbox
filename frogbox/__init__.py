"""
.. include:: ./intro.md
"""

__version__ = "0.3.0"

from .pipelines.supervised import SupervisedPipeline  # noqa: F401
from .pipelines.gan import GANPipeline  # noqa: F401
from .config import read_json_config  # noqa: F401
from .utils import load_model_checkpoint  # noqa: F401
from ignite.engine import Events  # noqa: F401
