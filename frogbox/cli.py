from pathlib import Path
import click


@click.group()
def cli():
    """
    An opinionated machine learning framework.
    """


@cli.group()
def config():
    """Manage configuration files."""


@config.command(name="new")
@click.option(
    "--type",
    "-t",
    "type_",
    type=click.Choice(["supervised"]),
    default="supervised",
    help="Pipeline type",
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
    "--output",
    "-o",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Save config to path.",
)
def new_config(type_: str, format_: str, output: Path):
    import json
    from .config import (
        SupervisedConfig,
        ObjectDefinition,
        ModelDefinition,
    )

    if type_ == "supervised":
        config = SupervisedConfig(
            project="example",
            datasets={
                "train": ObjectDefinition(
                    object="datasets.example.ExampleDataset",
                    kwargs={},
                ),
            },
            model=ModelDefinition(
                object="models.example.ExampleModel",
                kwargs={},
            ),
        )

        cfg_json = config.model_dump_json(
            indent=2,
            exclude_none=True,
            exclude={
                "loaders": True,
                "callbacks": True,
                "checkpoints": {"__all__": {"mode": True}},
            },
        )
    else:
        raise ValueError(f"Unknown pipeline type {type_}.")

    if format_ == "yaml":
        import yaml

        cfg_data = yaml.safe_dump(
            json.loads(cfg_json),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    elif format_ == "json":
        cfg_data = cfg_json
    else:
        raise ValueError(f"Unknown file format \"{format_}\".")

    if output is not None:
        output.write_text(cfg_data)
    else:
        print(cfg_data)


if __name__ == "__main__":
    cli()
