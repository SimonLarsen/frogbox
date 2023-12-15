"""@private"""

from typing import Optional
from pathlib import Path
import json
import importlib.resources
import click
from .config import read_json_config, SupervisedConfig, ObjectDefinition


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
    type=click.Choice(["supervised"]),
    default="supervised",
    help="Pipeline type.",
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
        f"train_{type_}.py",
        "model.py",
        "dataset.py",
    ]

    template_outputs = [
        dir_ / "train.py",
        dir_ / "models" / "example.py",
        dir_ / "datasets" / "example.py",
    ]

    overwrite_checks = template_outputs + [dir_ / "configs" / "example.json"]

    # Check if files already exist
    if not overwrite:
        for path in overwrite_checks:
            if path.exists():
                raise RuntimeError("Project directory is not empty.")

    # Create folders and copy template files
    resource_files = importlib.resources.files("stort.data")
    for input_resource, output_path in zip(template_inputs, template_outputs):
        file_data = resource_files.joinpath(input_resource).read_text()
        output_path.parent.mkdir(exist_ok=True, parents=True)
        output_path.write_text(file_data)

    # Create config template
    if type_ == "supervised":
        config = SupervisedConfig(
            type="supervised",
            project="example",
            model=ObjectDefinition(
                class_name="models.example.ExampleModel"
            ),
            datasets={
                "train": ObjectDefinition(
                    class_name="datasets.example.ExampleDataset"
                ),
                "val": ObjectDefinition(
                    class_name="datasets.example.ExampleDataset"
                )
            }
        )
        config_json = config.model_dump_json(indent=2, exclude_none=True)
    else:
        raise RuntimeError(f"Unknown pipeline type {type_}.")

    output_path = dir_ / "configs" / "example.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(config_json)


@cli.group()
def config():
    """Work with config files."""
    pass


@config.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Config file path.",
)
def validate(path: Path):
    """Validate config file."""
    read_json_config(path)


@config.command()
@click.option(
    "--type",
    "-t",
    "type_",
    type=click.Choice(["supervised"]),
    default="supervised",
    help="Pipeline type.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Write schema to file.",
)
def schema(type_: str, out: Optional[Path] = None):
    if type_ == "supervised":
        schema = json.dumps(SupervisedConfig.model_json_schema(), indent=2)
    else:
        raise RuntimeError(f"Unknown pipeline type {type_}.")

    if out:
        out.write_text(schema)
    else:
        print(schema)


if __name__ == "__main__":
    cli()
