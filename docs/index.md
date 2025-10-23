---
hide:
- navigation
---

# frogbox

Frogbox is an opinionated PyTorch machine learning framework built for rapid prototyping and research.

## Features

* Experiments are defined using JSON files and support [jinja2](https://jinja.palletsprojects.com) templates.
* Flexible event system inspired by [Ignite](https://pytorch.org/ignite).
* Automatic experiment tracking. Currently only [Weights & Biases](https://wandb.ai/) is supported with other platforms planned.
* CLI tool for easy project management.
* Integrates [Accelerate](https://huggingface.co/docs/accelerate/index) to support automatic mixed precision (AMP) and distributed training.

## Installation

You can install frogbox via PyPI with the following command:

```
pip install frogbox
```

## Creating a new project

Create a new project using the `frogbox` CLI tool:


=== "YAML"

    ```
    frogbox project new -d . -f yaml
    ```

=== "JSON"

    ```
    frogbox project new -d . -f json
    ```

Implement your model in `models`, your dataset in `datasets` and your pipeline configuration in `configs`.

Before training, configure your training system with accelerate:

```
accelerate config
```

Then launch the training pipeline:

```
accelerate launch -m frogbox.run -c configs/config.json
```

See example projects in the `examples` folder on [GitHub](https://github.com/SimonLarsen/frogbox/tree/main/examples).
