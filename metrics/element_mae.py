import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from typing import Sequence, Union


class ElementMeanAbsoluteError(Metric):
    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_absolute_errors = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
        self._sum_of_absolute_errors += (
            absolute_errors.flatten(1).mean(1).sum().to(self._device)
        )
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_absolute_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "ElementMeanAbsoluteError must have at least one example"
                " before it can be computed."
            )
        return self._sum_of_absolute_errors.item() / self._num_examples
