from typing import List, Union, Dict, Any
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
        pipeline_model_name: str = "model",
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1.")
        
        self.decay = decay
        self.pipeline_model_name = pipeline_model_name

        parameters = self._get_parameters(pipeline)
        self.shadow_params = [
            p.clone().detach()
            for p in parameters
        ]

    def _get_parameters(self, pipeline: PipelineType) -> List[torch.nn.Parameter]:
        model = getattr(pipeline, self.pipeline_model_name)
        return list(model.parameters())

    def __call__(self, pipeline: PipelineType) -> None:
        parameters = self._get_parameters(pipeline)
        one_minus_decay = 1.0 - self.decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = s_param - param
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "pipeline_model_name": self.pipeline_model_name,
            "shadow_params": self.shadow_params,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        self.pipeline_model_name = state_dict["pipeline_model_name"]
        self.shadow_params = state_dict["shadow_params"]
