"""@private"""

import click
import importlib.resources
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
        "train.py",
        f"config_{type_}.json",
        "example_model.py",
        "example_dataset.py",
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
    resource_files = importlib.resources.files("stort.data")
    for input_resource, output_path in zip(template_inputs, template_outputs):
        file_data = resource_files.joinpath(input_resource).read_text()
        output_path.parent.mkdir(exist_ok=True, parents=True)
        output_path.write_text(file_data)


if __name__ == "__main__":
    cli()
