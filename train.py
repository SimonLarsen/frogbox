from typing import Optional, Sequence
from pathlib import Path
import argparse
import torch
from utils import read_json_config
from pipelines.supervised import train


def parse_arguments(
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, default="configs/example.json"
    )
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--logging",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments()

    config = read_json_config(args.config)
    device = torch.device(args.device)

    train(
        config=config,
        device=device,
        checkpoint=args.checkpoint,
        logging=args.logging,
    )
