from setuptools import setup, find_packages
import re
from pathlib import Path


if __name__ == "__main__":
    here = Path(__file__).parent

    # Read package version
    version = re.search(
        r'__version__ = "(.+?)"',
        (here / "frogbox" / "__init__.py").read_text("utf8"),
    ).group(1)

    # Read requirements from requirements.txt
    requirements = (
        (here / "requirements.txt").read_text("utf8").strip().split("\n")
    )

    setup(
        name="frogbox",
        description="An opinionated machine learning framework.",
        version=version,
        author="Simon J. Larsen",
        author_email="simonhffh@gmail.com",
        license="MIT",
        packages=find_packages(),
        include_package_data=True,
        install_requires=requirements,
        entry_points={"console_scripts": ["frogbox=frogbox.cli:cli"]},
    )
