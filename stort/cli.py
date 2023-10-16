"""@private"""

import click
import importlib.resources
from pathlib import Path
from .config import Config


@click.group()
def cli():
    """
    A simple Torch + Ignite trainer.
    """
    pass


@cli.group()
def project():
    """Manage project."""
    pass


@project.command()
@click.option(
    "--type",
    "-t",
    "type_",
    type=click.Choice(["default", "minimal", "full"]),
    default="default",
    help="Configuration template type.",
)
@click.option(
    "--dir",
    "-d",
    "dir_",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
    default=Path("."),
    help="Project root directory.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing files if present.",
)
def new(type_: str, dir_: Path, overwrite: bool = False):
    """Create a new project from template."""
    config_path = dir_ / "configs" / "example.json"
    train_py_path = dir_ / "train.py"
    if not overwrite and (config_path.exists() or train_py_path.exists()):
        raise RuntimeError("Project directory is not empty.")

    # Create project directory
    dir_.mkdir(exist_ok=True, parents=True)

    # Write config file
    config_path.parent.mkdir(exist_ok=True)

    if type_ == "minimal":
        config_data = importlib.resources.read_text(
            package="stort.data",
            resource="config_minimal.json",
        )
    elif type_ == "default":
        config_data = importlib.resources.read_text(
            package="stort.data",
            resource="config_default.json",
        )
    elif type_ == "full":
        config_data = importlib.resources.read_text(
            package="stort.data",
            resource="config_default.json",
        )
        config = Config.model_validate_json(config_data)
        config_data = config.model_dump_json(indent=4)
    else:
        raise RuntimeError(f"Unknown template type {type_}.")

    with config_path.open("w") as fp:
        fp.write(config_data)

    # Write train.py script
    train_py_data = importlib.resources.read_text(
        package="stort.data",
        resource="train.py",
    )
    with train_py_path.open("w") as fp:
        fp.write(train_py_data)


if __name__ == "__main__":
    cli()
