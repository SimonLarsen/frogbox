"""@private"""

import click
from importlib.resources import path as resource_path
from pathlib import Path


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

    template_inputs = [
        resource_path("stort.data", "train.py"),
        resource_path("stort.data", f"config_{type_}.json"),
        resource_path("stort.data", "example_model.py"),
        resource_path("stort.data", "example_dataset.py"),
    ]

    template_outputs = [
        dir_ / "train.py",
        dir_ / "configs" / "example.json",
        dir_ / "models" / "example.py",
        dir_ / "datasets" / "example.py",
    ]

    # Check if files already exist
    if not overwrite:
        for path in template_outputs:
            if path.exists():
                raise RuntimeError("Project directory is not empty.")

    # Create folders and copy template files
    for input_path, output_path in zip(template_inputs, template_outputs):
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with input_path.open("r") as fp:
            file_data = fp.read()

        with output_path.open("w") as fp:
            fp.write(file_data)


if __name__ == "__main__":
    cli()
