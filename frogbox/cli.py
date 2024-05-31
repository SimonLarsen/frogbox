"""@private"""

from typing import Optional, Tuple, Sequence
from pathlib import Path
import os
import json
import importlib.resources
import click


@click.group()
def cli():
    """
    An opinionated machine learning framework.
    """
    pass


@cli.group()
def project():
    """Manage project."""
    pass


@project.command(name="new")
@click.option(
    "--type",
    "-t",
    "type_",
    type=click.Choice(["supervised", "gan"]),
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

        config = SupervisedConfig(
            type="supervised",
            project="example",
            datasets={
                "train": ObjectDefinition(
                    class_name="datasets.example.ExampleDataset"
                ),
                "val": ObjectDefinition(
                    class_name="datasets.example.ExampleDataset"
                ),
            },
            model=ObjectDefinition(class_name="models.example.ExampleModel"),
        )
        config_json = config.model_dump_json(indent=2, exclude_none=True)
    elif type_ == "gan":
        from .config import GANConfig, ObjectDefinition

        config = GANConfig(
            type="gan",
            project="example",
            datasets={
                "train": ObjectDefinition(
                    class_name="datasets.example.ExampleDataset"
                ),
                "val": ObjectDefinition(
                    class_name="datasets.example.ExampleDataset"
                ),
            },
            model=ObjectDefinition(class_name="models.example.ExampleModel"),
            disc_model=ObjectDefinition(
                class_name="models.example.ExampleModel"
            ),
        )
        config_json = config.model_dump_json(indent=2, exclude_none=True)
    else:
        raise RuntimeError(f"Unknown pipeline type {type_}.")

    output_path = dir_ / "configs" / "example.json"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(config_json)


@cli.group()
def service():
    """Manage service."""
    pass


@service.command(name="new")
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
def new_service(dir_: Path, overwrite: bool = False):
    """Create new service from template."""

    template_inputs = [
        "service.py",
    ]

    template_outputs = [
        dir_ / "service.py",
    ]

    # Check if files already exist
    if not overwrite:
        for path in template_outputs:
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


@service.command(name="serve")
@click.option(
    "--checkpoint",
    "-c",
    "checkpoints",
    type=(
        str,
        click.Path(
            exists=True, file_okay=True, dir_okay=False, path_type=Path
        ),
    ),
    multiple=True,
    help=(
        "Add model checkpoint."
        " Add multiple models by repeating this argument."
    ),
    metavar="NAME PATH",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cpu",
    help="CUDA device.",
    show_default=True,
)
def serve(checkpoints: Sequence[Tuple[str, Path]], device: str):
    """Serve service locally."""

    import uvicorn

    checkpoints_env = {}
    for name, path in checkpoints:
        checkpoints_env[name] = str(path)
    os.environ["CHECKPOINTS"] = json.dumps(checkpoints_env)
    os.environ["DEVICE"] = device

    uvicorn.run("service:app", port=8000, app_dir=".")


@service.command(name="dockerfile")
@click.option(
    "--checkpoint",
    "-c",
    "checkpoints",
    type=(
        str,
        click.Path(
            exists=True, file_okay=True, dir_okay=False, path_type=Path
        ),
    ),
    multiple=True,
    help=(
        "Add model checkpoint."
        " Add multiple models by repeating this argument."
    ),
    metavar="NAME PATH",
)
@click.option(
    "--requirements",
    "-r",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=Path
    ),
    default="requirements.txt",
    help="Path to service requirements.txt. Defaults to requirements.txt.",
    metavar="PATH",
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
    help="Write Dockerfile to file.",
)
def service_dockerfile(
    checkpoints: Sequence[Tuple[str, Path]],
    requirements: Path,
    out: Optional[Path] = None,
):
    """Build service Dockerfile."""
    from jinja2 import Environment, PackageLoader

    env = Environment(
        loader=PackageLoader("frogbox", "data"),
        autoescape=False,
    )
    template = env.get_template("Dockerfile")

    ckpt_info = []
    for name, model_path in checkpoints:
        config_path = model_path.parent / "config.json"
        ckpt_info.append(
            dict(
                name=name,
                model_path=model_path,
                config_path=config_path,
                parent_path=model_path.parent,
            )
        )

    env_checkpoints = {e["name"]: str(e["model_path"]) for e in ckpt_info}
    output = template.render(
        checkpoints=ckpt_info,
        requirements=str(requirements),
        env_checkpoints=json.dumps(env_checkpoints),
    )

    if out:
        out.write_text(output)
    else:
        print(output)


@cli.group()
def config():
    """Work with config files."""
    pass


@config.command()
@click.option(
    "--file",
    "-f",
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
    from .config import read_json_config

    read_json_config(path)


@config.command()
@click.option(
    "--type",
    "-t",
    "type_",
    type=click.Choice(["supervised", "gan"]),
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
        from .config import SupervisedConfig

        schema = json.dumps(SupervisedConfig.model_json_schema(), indent=2)
    elif type_ == "gan":
        from .config import GANConfig

        schema = json.dumps(GANConfig.model_json_schema(), indent=2)
    else:
        raise RuntimeError(f"Unknown pipeline type {type_}.")

    if out:
        out.write_text(schema)
    else:
        print(schema)


if __name__ == "__main__":
    cli()
