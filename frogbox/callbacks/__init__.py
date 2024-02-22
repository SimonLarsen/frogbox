"""
# Custom callbacks

Custom callbacks can be created by implementing a function that accepts the pipeline as its only argument.

For instance, in the following example a callback is added to unfreeze the model's encoder after 20 epochs:

```python
from frogbox import Events

def unfreeze_encoder(pipeline)
    model = pipeline.model
    model.encoder.requires_grad_(True)

pipeline.install_callback(
    event=Events.EPOCH_STARTED(once=20),
    callback=unfreeze_encoder,
)
```
"""
from .image_logger import create_image_logger  # noqa: F401
