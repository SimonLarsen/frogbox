frogbox
=======

![MIT License](https://img.shields.io/badge/license-MIT%20License-blue.svg)

An opinionated machine learning framework.

The workflow is built around [Torch Ignite](https://pytorch-ignite.ai) and uses [Weights & Biases](https://wandb.ai) for logging metrics.

Experiments are defined using JSON files and supports [jinja2](https://jinja.palletsprojects.com) templates. See `frogbox.config.Config` for configuration file JSON schema.

## Installation

```sh
pip install git+https://SimonLarsen@github.com/SimonLarsen/frogbox.git
```

## Getting started

Create a new project using the `frogbox` CLI tool:

```sh
frogbox project new -t supervised -d .
```

Implement your model, dataset and pipeline configuration. Then train the model by `train.py`:

```sh
frogbox train.py -c configs/myconfig.json -d cuda:0
```

See example project in `example/train.py`.

## Guides

* [Loading a trained model](frogbox/utils.html#loading-a-trained-model)
* [Logging images to W&B](frogbox/callbacks/image_logger.html#logging-images)
* [Custom callbacks](frogbox/callbacks.html#custom-callbacks)
