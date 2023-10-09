stort
=====

A simple Torch + Ignite trainer.

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
