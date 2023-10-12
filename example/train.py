from typing import Optional, Sequence
from pathlib import Path
import argparse
import torch
from stort import read_json_config, train_supervised
from stort.callbacks import create_image_logger_callback


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

    dataset_params = config.datasets["test"].params
    image_logger = create_image_logger_callback(
        normalize_mean=dataset_params["normalize_mean"],
        normalize_std=dataset_params["normalize_std"],
        denormalize_input=dataset_params["do_normalize"],
    )

    train_supervised(
        config=config,
        device=args.device,
        checkpoint=args.checkpoint,
        logging=args.logging,
        callbacks=[image_logger],
    )
