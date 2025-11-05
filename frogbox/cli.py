from pathlib import Path
import importlib.resources
import click


@click.group()
def cli():
    """
    An opinionated machine learning framework.
    """


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
    "--format",
    "-f",
    "format_",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Config file format.",
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
def new_project(type_: str, format_: str, dir_: Path, overwrite: bool = False):
    """Create a new project from template."""
    import json
    from .config import (
        ConfigType,
        SupervisedConfig,
        ObjectDefinition,
        ModelDefinition,
    )

    template_inputs = [
        "model.py",
        "dataset.py",
    ]

    template_outputs = [
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
    example_model = ModelDefinition(
        object="models.example.ExampleModel",
    )
    example_dataset = ObjectDefinition(
        object="datasets.example.ExampleDataset"
    )

    if type_ == "supervised":
        cfg = SupervisedConfig(
            type=ConfigType.SUPERVISED,
            project="example",
            tracker="wandb",
            model=example_model,
            datasets={
                "train": example_dataset,
                "val": example_dataset,
            },
        )
    else:
        raise RuntimeError(f"Unknown pipeline type {type_}.")

    cfg_json = cfg.model_dump_json(indent=4, exclude_none=True)
    if format_ == "yaml":
        import yaml

        cfg_data = yaml.safe_dump(json.loads(cfg_json))
        output_path = dir_ / "configs" / "example.yaml"
    elif format_ == "json":
        cfg_data = cfg_json
        output_path = dir_ / "configs" / "example.json"
    else:
        raise ValueError(f"Unknown file format \"{format_}\".")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(cfg_data)


if __name__ == "__main__":
    cli()
