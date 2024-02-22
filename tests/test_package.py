from packaging.version import Version


def test_import():
    import frogbox
    Version(frogbox.__version__)
