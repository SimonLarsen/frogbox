from typing import Optional, Union, Dict, Any
import copy
import torch
from ..pipelines.supervised import SupervisedPipeline
from ..pipelines.gan import GANPipeline


PipelineType = Union[SupervisedPipeline, GANPipeline]


class EMACallback:
    def __init__(
        self,
        pipeline: PipelineType,
        decay: float,
        handle_buffers: Optional[str] = "copy",
        pipeline_model_name: str = "model",
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1.")
        if handle_buffers is not None and handle_buffers not in (
            "update",
            "copy",
        ):
            raise ValueError("`handle_buffers` must be in ('update', 'copy').")

        self.decay = decay
        self.handle_buffers = handle_buffers
        self.pipeline_model_name = pipeline_model_name

        model = self._get_pipeline_model(pipeline)
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.detach_()
        self.ema_model.eval()

    def _get_pipeline_model(self, pipeline: PipelineType) -> torch.nn.Module:
        model = getattr(pipeline, self.pipeline_model_name)
        return model

    def __call__(self, pipeline: PipelineType) -> None:
        model = self._get_pipeline_model(pipeline)
        for ema_p, model_p in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_p.mul_(self.decay).add_(model_p.data, alpha=(1.0 - self.decay))

        if self.handle_buffers == "update":
            for ema_b, model_b in zip(
                self.ema_model.buffers(), model.buffers()
            ):
                try:
                    ema_b.mul_(self.decay).add_(
                        model_b.data, alpha=(1 - self.decay)
                    )
                except RuntimeError:
                    ema_b.data = model_b.data
        elif self.handle_buffers == "copy":
            for ema_b, model_b in zip(
                self.ema_model.buffers(), model.buffers()
            ):
                ema_b.data = model_b.data

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "handle_buffers": self.handle_buffers,
            "pipeline_model_name": self.pipeline_model_name,
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        self.handle_buffers = state_dict["handle_buffers"]
        self.pipeline_model_name = state_dict["pipeline_model_name"]
        self.ema_model.load_state_dict(state_dict["ema_model"])
