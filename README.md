stort
=====

![MIT License](https://img.shields.io/badge/license-MIT%20License-blue.svg)

An opinionated framework for training and serving machine learning models.

The workflow is built around [Torch Ignite](https://pytorch-ignite.ai) and uses [Weights & Biases](https://wandb.ai) for logging metrics.

Experiments are defined using JSON files and supports [jinja2](https://jinja.palletsprojects.com) templates. See `stort.config.Config` for configuration file JSON schema.

## Installation

```sh
pip install git+https://SimonLarsen@github.com/SimonLarsen/stort.git
```

## Getting started

Create a new project using the `stort` CLI tool:

```sh
stort project new -t supervised -d .
```

Implement your model, dataset and pipeline configuration. Then train the model by `train.py`:

```sh
stort train.py -c configs/myconfig.json -d cuda:0
```

See example project in `example/train.py`.

## Guides

* [Loading a trained model](stort/utils.html#loading-a-trained-model)
* [Logging images to W&B](stort/callbacks/image_logger.html#logging-images)
* [Custom callbacks](stort/callbacks.html#custom-callbacks)
