# Implementing custom callbacks

Custom callbacks are created by implementing a function that takes in a pipeline as its only argument and returns nothing. 

In this example, we define two callbacks: one that freezes the weights of the model's encoder and one that unfreezes them.


```python title="callbacks/encoder.py"
from frogbox import SupervisedPipeline

def freeze_encoder(pipeline: SupervisedPipeline):
    model = pipeline.model
    model.encoder.requires_grad_(False)

def unfreeze_encoder(pipeline: SupervisedPipeline):
    model = pipeline.model
    model.encoder.requires_grad_(True)
```

Callbacks are installed in the pipeline by adding them to the `callbacks` field in the pipeline configuration file.

In this example, we run `freeze_encoder` at the very beginning and then run `unfreeze_encoder` once after 20 epochs.

```yaml title="configs/config.yml"
callbacks:
- function: callbacks.encoder.freeze_encoder
  interval: started

- function: callbacks.encoder.unfreeze_encoder
  interval:
    event: epoch_completed
    first: 20
    last: 20
```