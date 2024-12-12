<p align="center">
    <img src="https://simonlarsen.github.io/frogbox/logo.png" width="224">
</p>
<h1 align="center">
    frogbox
</h1>

Frogbox is an opinionated machine learning framework for PyTorch built for rapid prototyping and research.

## Features

* Experiments are defined using JSON files and support [jinja2](https://jinja.palletsprojects.com) templates.
* Automatic experiment tracking. Currently only [Weights & Biases](https://wandb.ai/) is supported with other platforms planned.
* CLI tool for easy project management. Just type `frogbox project new -t supervised` to get started.
* Integrates [Accelerate](https://huggingface.co/docs/accelerate/index) to support automatic mixed precision (AMP) and distributed training.

## Installation

```sh
pip install git+https://SimonLarsen@github.com/SimonLarsen/frogbox.git@v0.5.0
```

## Getting started

See [documentation](https://simonlarsen.github.io/frogbox).
