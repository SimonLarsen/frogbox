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

## Getting started

Create a new project using the `stort` CLI tool:

```sh
stort project new -t default -d myproject
```

See example project in [example/train.py](example/train.py).

## Loading a trained model

Trained models can be loaded with `stort.utils.load_model_checkpoint`. The function returns the trained model as well the trainer configuration.

```python
import torch
from stort.utils import load_model_checkpoint

model, config = load_model_checkpoint("checkpoints/mymodel/checkpoint.py")

device = torch.device("cuda:0")
model = model.eval().to(device)

x = torch.rand((1, 3, 16, 16), device=device)
with torch.inference_mode():
    pred = model(x)
```

## Logging images

The simplest way to log images during training is to create an callback with `stort.callbacks.image_logger.create_image_logger`:

```python
from stort import train_supervised
from stort.callbacks import create_image_logger

train_supervised(
    config=config,
    device=device,
    checkpoint=checkpoint,
    callbacks=[create_image_logger()]
)
```

Images can automatically be denormalized by setting `denormalize_input`/`denormalize_output` and providing the mean and standard deviation used for normalization.

For instance, if input images are normalized with ImageNet parameters and outputs are in [0, 1]:

```python
image_logger = create_image_logger(
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    denormalize_input=True,
)
```

More advanced transformations can be made by overriding `input_transform`, `model_transform`, or `output_transform`:

```python
from torchvision.transforms.functional import hflip

def flip_input(x, y, y_pred):
    x = hflip(x)
    return x, y_pred, y

image_logger = create_image_logger(
    output_transform=flip_input,
)
```

## Callbacks

Custom callbacks can be added by creating a `stort.callbacks.Callback` object.

For instance, in the following example a callback is added to unfreeze the model's encoder after 20 epochs:

```python
from stort import Events
from stort.callbacks import Callback, CallbackState

def unfreeze_encoder(state: CallbackState)
    state.model.encoder.requires_grad_(True)

unfreeze_callback = Callback(Events.EPOCH_STARTED(once=20), unfreeze_encoder)

train_supervised(
    ...
    callbacks=[unfreeze_callback]
)
```

## JSON config validation

A JSON schema can be generated to use for validation by calling `stort config schema`.

### Schema validation and completion in Visual Studio Code

To enable schema validation in VSCode generate a schema and add it to your `settings.json` under `json.schemas`.

**Warning: This overrides your current workspace settings.**

```sh
stort config schema -o .config_schema
cat > .vscode/settings.json << EOL
{
    "json.schemas": [
        {
            "fileMatch": [
                "/configs/*.json"
            ],
            "url": "./.config_schema"
        }
    ]
}
EOL
```
