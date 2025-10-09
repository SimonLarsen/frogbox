from typing import cast, Mapping, Optional, Sequence
from pathlib import Path
import click
from frogbox.config import read_config
from frogbox.pipelines.pipeline import Pipeline
from frogbox import (
    SupervisedConfig,
    SupervisedPipeline,
)


def _validate_vars(ctx, param, values) -> Mapping[str, str]:
    out = {}
    for value in values:
        pos = value.find("=")
        assert pos >= 1
        out[value[:pos]] = value[pos + 1 :]
    return out


def _validate_checkpoint_keys(ctx, param, values) -> Sequence[str]:
    out = []
    for value in values:
        for key in value.split(","):
            out.append(key.strip())
    return out


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Config file.",
)
@click.option(
    "--config-format",
    "-f",
    type=click.Choice(["yaml", "json"]),
    help="Config file format.",
)
@click.option(
    "--checkpoint",
    type=Path,
    help="Path to checkpoint.",
)
@click.option(
    "--checkpoint-keys",
    type=str,
    multiple=True,
    callback=_validate_checkpoint_keys,
    help="Keys in checkpoint to load. Uses all keys if not provided.",
)
@click.option(
    "--var",
    "-v",
    "config_vars",
    type=str,
    multiple=True,
    callback=_validate_vars,
    help="Set config parser variable.",
)
def run(
    config: Path,
    config_format: Optional[str] = None,
    checkpoint: Optional[Path] = None,
    checkpoint_keys: Optional[Sequence[str]] = None,
    config_vars: Optional[Mapping[str, str]] = None,
    **kwargs,
) -> Pipeline:
    if config_vars is None:
        config_vars = {}

    if checkpoint_keys is not None and len(checkpoint_keys) == 0:
        checkpoint_keys = None

    cfg = read_config(
        path=config,
        format=config_format,
        config_vars=config_vars,
    )

    if cfg.type == "supervised":
        cfg = cast(SupervisedConfig, cfg)
        pipeline = SupervisedPipeline(
            config=cfg,
            checkpoint=checkpoint,
            checkpoint_keys=checkpoint_keys,
            **kwargs,
        )
    else:
        raise RuntimeError(f'Unknown config type "{cfg.type}".')
    pipeline.run()
    return pipeline


if __name__ == "__main__":
    run()
