from pathlib import Path
from typing import cast

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from ..pipelines.supervised import SupervisedPipeline
from ..utils import fix_compiled_model_keys

first_call: bool = True
ema_model: AveragedModel | None


def ema_update(
    pipe: SupervisedPipeline,
    decay: float = 0.999,
    use_buffers: bool = False,
) -> None:
    """
    Update exponential moving average (EMA) model.

    First call initializes the EMA model.

    Parameters
    ----------
    decay
        Decay rate for EMA. Must be in the range [0, 1].
    use_buffers
        If `True`, running averages will also be computed for buffers.
    """
    global first_call, ema_model

    if first_call:
        first_call = False
        ema_model = AveragedModel(
            model=pipe.model,
            device=pipe.device,
            multi_avg_fn=get_ema_multi_avg_fn(decay),
            use_buffers=use_buffers,
        )

    ema_model = cast(AveragedModel, ema_model)
    ema_model.update_parameters(pipe.model)


def ema_save(
    pipe: SupervisedPipeline,
    filename: str = "ema.pt",
) -> None:
    """
    Save EMA model.

    [`ema_update`][frogbox.callbacks.ema.ema_update] must be called before [`ema_save`][frogbox.callbacks.ema.ema_save].

    Parameters
    ----------
    filename
        Filename to save model static dict to.
    """

    if first_call or ema_model is None:
        raise RuntimeError("EMA model is not initialized.")

    output_folder = Path("checkpoints") / pipe.run_name
    path = output_folder / filename

    output_folder.mkdir(parents=True, exist_ok=True)

    model = pipe.accelerator.unwrap_model(pipe.model)
    state_dict = fix_compiled_model_keys(model.state_dict())

    torch.save(state_dict, path)
