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

    normalize_mean = config.datasets["test"].params["normalize_mean"]
    normalize_std = config.datasets["test"].params["normalize_std"]

    def image_logger_output_transform(x, y, y_pred):
        # Denormalize input image
        mean = torch.as_tensor(
            normalize_mean,
            dtype=x.dtype,
            device=x.device,
        ).reshape((3, 1, 1))
        std = torch.as_tensor(
            normalize_std,
            dtype=x.dtype,
            device=x.device,
        ).reshape((3, 1, 1))
        x = x * std + mean
        # Draw order input, output, target
        return x, y_pred, y

    image_logger = create_image_logger_callback(
        output_transform=image_logger_output_transform,
    )

    train_supervised(
        config=config,
        device=args.device,
        checkpoint=args.checkpoint,
        logging=args.logging,
        callbacks=[image_logger],
    )
