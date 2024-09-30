# Getting started

Create a new project using the `frogbox` CLI tool:

```sh
frogbox project new -t supervised -d .
```

Implement your model in `models`, your dataset in `datasets` and your pipeline configuration in `configs`.

If you need to use distributed training or have a system with more than one CUDA device,
configure your training system:

```sh
frogbox config
```

Then launch the training pipeline:

```sh
frogbox launch train.py -c configs/config.json
```

See example projects in `examples`.

# Guides

* [Loading a trained model](frogbox/utils.html#loading-a-trained-model)
* [Logging images to W&B](frogbox/callbacks/image_logger.html#logging-images)
* [Custom callbacks](frogbox/callbacks.html#custom-callbacks)
