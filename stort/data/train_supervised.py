from typing import Optional, Sequence
from pathlib import Path
import argparse
import torch
from stort import read_json_config, SupervisedPipeline


def parse_arguments(
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, default="configs/example.json"
    )
    parser.add_argument(
        "-d", "--device", type=torch.device, default=torch.device("cuda:0")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-keys", type=str, nargs="+")
    parser.add_argument(
        "--logging",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-id", type=str, required=False)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments()
    config = read_json_config(args.config)

    pipeline = SupervisedPipeline(
        config=config,
        device=args.device,
        checkpoint=args.checkpoint,
        checkpoint_keys=args.checkpoint_keys,
        logging=args.logging,
        wandb_id=args.wandb_id,
    )

    pipeline.run()
