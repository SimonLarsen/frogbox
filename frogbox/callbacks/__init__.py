"""
# Custom callbacks

Custom callbacks can be created by implementing a function that accepts the pipeline as its first argument.

For instance, in the following example a callback is added to unfreeze the model's encoder after 20 epochs:

```python
from frogbox import Event

def unfreeze_encoder(pipeline)
    model = pipeline.model
    model.encoder.requires_grad_(True)

pipeline.install_callback(
    event=Event("epoch_started", first=20, last=20),
    callback=unfreeze_encoder,
)
```
"""  # noqa: E501

from .image_logger import ImageLogger  # noqa: F401
