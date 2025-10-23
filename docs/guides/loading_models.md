# Loading a trained model

Trained models can be loaded from a checkpoint using `load_model_checkpoint`:

```python
from frogbox.utils import load_model_checkpoint

model, config = load_model_checkpoint("checkpoint/cool-giraffe-123/checkpoint_1234.pt")
model = model.eval()

with torch.inference_mode():
    pred = model(torch.randn(1, 3, 128, 128))
```

::: frogbox.utils
    options:
      members:
        - load_model_checkpoint