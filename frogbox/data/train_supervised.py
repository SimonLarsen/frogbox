from typing import cast, Optional, Sequence
from pathlib import Path
import argparse
from frogbox import read_json_config, SupervisedPipeline, SupervisedConfig


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
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments()
    config = cast(SupervisedConfig, read_json_config(args.config))

    pipeline = SupervisedPipeline(
        config=config,
        checkpoint=args.checkpoint,
        checkpoint_keys=args.checkpoint_keys,
        logging=args.logging,
        wandb_id=args.wandb_id,
        tags=args.tags,
        group=args.group,
    )

    pipeline.run()
