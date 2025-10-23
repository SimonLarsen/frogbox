# Metrics

Metrics are used to evaluate model performance.
Frogbox comes with a wide range of metrics through [torchmetrics](https://lightning.ai/docs/torchmetrics/).

For instance, to use SSIM to evaluate a supervised model, one can add the
SSIM metric from torchmetrics like this:

```yaml
metrics:
  SSIM:
    object: torchmetrics.image.StructuralSimilarityIndexMeasure
    kwargs:
      data_range: 1.0
      kernel_size: 11
      sigma: 1.5
```

## Implementing custom metrics

Custom metrics must subclass `torchmetrics.Metric`.

See [this guide](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html)
for a detailed description of how to implement custom metrics.
