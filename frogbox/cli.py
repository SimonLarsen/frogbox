"""@private"""

from pathlib import Path
import os
import shutil
import importlib
import subprocess
import click


@click.group()
def cli():
    """
    An opinionated machine learning framework.
    """


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[],
    )
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def launch(args):
    """
    Launch script.

    Alias for `accelerate launch`.
    """
    exec_path = shutil.which("accelerate")
    cmd = [exec_path, "launch"] + list(args)
    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=False)


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        help_option_names=[],
    )
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def config(args):
    """
    Configure training system.

    Alias for `accelerate config`.
    """
    exec_path = shutil.which("accelerate")
    cmd = [exec_path, "config"] + list(args)
    env = os.environ.copy()
    subprocess.run(cmd, env=env, check=False)


@cli.group()
def project():
    """Manage project."""


@project.command(name="new")
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
def new_project(type_: str, dir_: Path, overwrite: bool = False):
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
                raise RuntimeError(
                    f"File '{path}' already exists."
                    " Use flag --overwrite to overwrite."
                )

    # Create folders and copy template files
    resource_files = importlib.resources.files("frogbox.data")
    for input_resource, output_path in zip(template_inputs, template_outputs):
        file_data = resource_files.joinpath(input_resource).read_text()
        output_path.parent.mkdir(exist_ok=True, parents=True)
        output_path.write_text(file_data)

    # Create config template
    if type_ == "supervised":
        from .config import SupervisedConfig, ObjectDefinition

        config_json = SupervisedConfig(
            type="supervised",
            project="example",
            model=ObjectDefinition(class_name="models.example.ExampleModel"),
            datasets={
                "train": ObjectDefinition(
                    class_name="datasets.example.ExampleDataset"
                ),
                "val": ObjectDefinition(
                    class_name="datasets.example.ExampleDataset"
                ),
            },
        ).model_dump_json(indent=4, exclude_none=True)
    else:
        raise RuntimeError(f"Unknown pipeline type {type_}.")

    output_path = dir_ / "configs" / "example.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(config_json)


if __name__ == "__main__":
    cli()
