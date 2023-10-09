from setuptools import setup, find_packages
import re
from pathlib import Path


here = Path(__file__).parent
version = re.search(
    r'__version__ = "(.+?)"',
    (here / "stort" / "__init__.py").read_text("utf8"),
).group(1)

setup(
    name="stort",
    description="A simple Torch trainer.",
    version=version,
    author="Simon J. Larsen",
    author_email="simonhffh@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0,<2.1.0",
        "torchvision>=0.15.0,<0.16.0",
        "pytorch-ignite>=0.4.12",
        "wandb>=0.15.12",
        "tqdm>=4.66.1",
    ],
)
