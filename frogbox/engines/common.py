from typing import Optional
import torch


def _backward_with_scaler(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    iteration: int,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
):
    if scaler:
        scaler.scale(loss).backward()
        if iteration & gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=clip_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
    else:
        loss.backward()
        if iteration % gradient_accumulation_steps == 0:
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=clip_grad_norm,
                )
            optimizer.step()
