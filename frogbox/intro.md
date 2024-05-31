# Getting started

Create a new project using the `frogbox` CLI tool:

```sh
frogbox project new -t supervised -d .
```

Implement your model in `models`, your dataset in `datasets` and your pipeline configuration in `configs`.
Then train the model by running `train.py`:

```sh
frogbox train.py -c configs/config.json -d cuda:0
```

See example project in `examples/supervised`.

# Guides

* [Loading a trained model](frogbox/utils.html#loading-a-trained-model)
* [Logging images to W&B](frogbox/callbacks/image_logger.html#logging-images)
* [Custom callbacks](frogbox/callbacks.html#custom-callbacks)