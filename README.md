stort
=====

![MIT License](https://img.shields.io/badge/license-MIT%20License-blue.svg)

A simple Torch + Ignite trainer.

The workflow is built around [Torch Ignite](https://pytorch-ignite.ai) and uses [Weights & Biases](https://wandb.ai) for logging metrics.

Experiments are defined using JSON files and supports [jinja2](https://jinja.palletsprojects.com) templates. See `stort.config.Config` for configuration file JSON schema.

## Installation

```sh
pip install git+https://SimonLarsen@github.com/SimonLarsen/torch-ignite-template.git
```

## Usage

```python
import torch
from stort import read_json_config, train_supervised

config = read_json_config("config.json")
train_supervised(config=config, device=torch.device("cuda:0"))
```

See example project in `example/train.py`.
