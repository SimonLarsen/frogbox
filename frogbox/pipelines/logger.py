from typing import Any, Callable, List, Optional, Union
from torch.optim import Optimizer
from ignite.engine import Engine, Events
from ignite.handlers.base_logger import (
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
)
from accelerate import Accelerator


class AccelerateLogger(BaseLogger):
    def __init__(self, accelerator: Accelerator):
        self._accelerator = accelerator

    def _create_output_handler(
        self, *args: Any, **kwargs: Any
    ) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(
        self, *args: Any, **kwargs: Any
    ) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)


class OutputHandler(BaseOutputHandler):
    def __init__(
        self,
        tag: str,
        metric_names: Optional[List[str]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[
            Callable[[Engine, Union[str, Events]], int]
        ] = None,
        state_attributes: Optional[List[str]] = None,
    ):
        super().__init__(
            tag,
            metric_names,
            output_transform,
            global_step_transform,
            state_attributes,
        )

    def __call__(
        self,
        engine: Engine,
        logger: AccelerateLogger,
        event_name: Union[str, Events],
    ) -> None:
        if not isinstance(logger, AccelerateLogger):
            raise RuntimeError(
                f"Handler '{self.__class__.__name__}'"
                " works only with AccelerateLogger."
            )

        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        metrics = self._setup_output_metrics_state_attrs(
            engine, log_text=True, key_tuple=False
        )
        logger._accelerator.log(metrics, step=global_step)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    def __init__(
        self,
        optimizer: Optimizer,
        param_name: str = "lr",
        tag: Optional[str] = None,
    ):
        super(OptimizerParamsHandler, self).__init__(
            optimizer, param_name, tag
        )

    def __call__(
        self,
        engine: Engine,
        logger: AccelerateLogger,
        event_name: Union[str, Events],
    ) -> None:
        if not isinstance(logger, AccelerateLogger):
            raise RuntimeError(
                "Handler OptimizerParamsHandler works"
                " only with AccelerateLogger."
            )

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        params = {
            f"{tag_prefix}{self.param_name}/group_{i}": float(
                param_group[self.param_name]
            )
            for i, param_group in enumerate(self.optimizer.param_groups)
        }
        logger._accelerator.log(params, step=global_step)
