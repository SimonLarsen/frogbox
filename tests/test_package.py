from packaging.version import Version


def test_import():
    import stort
    Version(stort.__version__)
