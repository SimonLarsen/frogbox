from typing import cast, Optional, Sequence
from pathlib import Path
import argparse
from frogbox import read_json_config, SupervisedPipeline, SupervisedConfig
from frogbox.callbacks import ImageLogger


def parse_arguments(
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, default="configs/example.json"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-keys", type=str, nargs="+")
    parser.add_argument(
        "--logging",
        type=str,
        choices=["online", "offline"],
        default="online",
    )
    parser.add_argument("--wandb-id", type=str, required=False)
    parser.add_argument("--tags", type=str, nargs="+")
    parser.add_argument("--group", type=str)
    parser.add_argument("-v", "--var", type=str, action="append", default=[])
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments()
    config = cast(SupervisedConfig, read_json_config(args.config, *args.var))

    pipeline = SupervisedPipeline(
        config=config,
        checkpoint=args.checkpoint,
        checkpoint_keys=args.checkpoint_keys,
        logging=args.logging,
        wandb_id=args.wandb_id,
        tags=args.tags,
        group=args.group,
    )

    ds_conf = config.datasets["train"].params
    image_logger = ImageLogger(
        denormalize_input=True,
        normalize_mean=ds_conf["normalize_mean"],
        normalize_std=ds_conf["normalize_std"],
    )
    pipeline.install_callback(pipeline.log_interval, image_logger)

    pipeline.run()
