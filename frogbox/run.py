from typing import cast, Mapping, Optional
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
    help="Config file.",
)
@click.option(
    "--var",
    "-v",
    "config_vars",
    type=str,
    multiple=True,
    callback=_validate_vars,
)
def run(
    config: Path,
    config_vars: Optional[Mapping[str, str]] = None,
    **kwargs,
) -> Pipeline:
    if config_vars is None:
        config_vars = {}

    cfg = read_config(config, config_vars=config_vars)
    if cfg.type == "supervised":
        cfg = cast(SupervisedConfig, cfg)
        pipeline = SupervisedPipeline(cfg, **kwargs)
    else:
        raise RuntimeError(f'Unknown config type "{cfg.type}".')
    pipeline.run()
    return pipeline


if __name__ == "__main__":
    run()
