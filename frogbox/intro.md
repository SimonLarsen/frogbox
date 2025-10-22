# Getting started

Create a new project using the `frogbox` CLI tool:

```sh
frogbox project new -d . -f yaml
```

Implement your model in `models`, your dataset in `datasets` and your pipeline configuration in `configs`.

Before training, configure your training system with accelerate:

```sh
accelerate config
```

Then launch the training pipeline:

```sh
accelerate launch -m frogbox.run -c configs/config.json
```

See example projects in `examples`.

# Guides

* [Using and implementing metrics](frogbox/metrics.html)
* [Loading a trained model](frogbox/utils.html#loading-a-trained-model)
* [Logging images to W&B](frogbox/callbacks/image_logger.html#logging-images)
* [Custom callbacks](frogbox/callbacks.html#custom-callbacks)
