from packaging.version import Version
import importlib


def test_import():
    import frogbox
    Version(frogbox.__version__)


def test_import_dist_versions():
    import frogbox
    assert frogbox.__version__ == importlib.metadata.version("frogbox")
