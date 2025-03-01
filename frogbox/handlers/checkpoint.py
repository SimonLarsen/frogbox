from typing import (
    Mapping,
    Any,
    Callable,
    Sequence,
    Optional,
    List,
    Dict,
)
import os
import math
from pathlib import Path
from collections import namedtuple
import torch
from accelerate import Accelerator
from accelerate.utils.other import is_compiled_module
from ..config import Config


SavedCheckpoint = namedtuple("SavedCheckpoint", ["filename", "priority"])


class Checkpoint:
    def __init__(
        self,
        accelerator: Accelerator,
        config: Config,
        to_save: Mapping[str, Any],
        output_folder: str | os.PathLike,
        global_step_function: Callable[[], int],
        score_function: Optional[Callable[[], float]] = None,
        score_name: Optional[str] = None,
        score_mode: str = "max",
        to_unwrap: Optional[Sequence[str]] = None,
        filename_prefix: str = "checkpoint",
        max_saved: int = 3,
    ):
        assert score_mode in ("min", "max")

        self._accelerator = accelerator
        self._config = config
        self._to_save = to_save
        self._output_folder = output_folder
        self._global_step_function = global_step_function
        self._score_function = score_function
        self._score_name = score_name
        self._score_mode = score_mode
        self._filename_prefix = filename_prefix
        self._max_saved = max_saved
        if to_unwrap is None:
            to_unwrap = []
        self._to_unwrap = to_unwrap

        self._saved: List[SavedCheckpoint] = []

    def _get_filename(
        self,
        step: Optional[int],
        score: Optional[float],
    ) -> str:
        name = str(Path(self._output_folder) / self._filename_prefix)

        if step is not None:
            name += "_" + str(step)

        if score is not None:
            if self._score_name is not None:
                name += f"_{self._score_name}={score:.4f}"
            else:
                name += f"_{score:.4f}"

        name += ".pt"
        return name

    def _save_checkpoint(
        self,
        filename: str,
    ) -> None:
        if not self._accelerator.is_local_main_process:
            return

        # Create parent directory if it doesn't exist
        parent_dir = Path(filename).parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Save config file
        config_json = self._config.model_dump_json(indent=4, exclude_none=True)
        with (parent_dir / "config.json").open("w") as fp:
            fp.write(config_json)

        # Extract state dicts from objects
        state_dicts = {}
        for key, obj in self._to_save.items():
            if key in self._to_unwrap:
                obj = self._accelerator.unwrap_model(obj)
                if is_compiled_module(obj):
                    obj = obj._orig_mod
            state_dicts[key] = obj.state_dict()

        torch.save(state_dicts, filename)

    def __call__(self) -> None:
        # Compute checkpoint score/priority
        score = None
        priority = 0.0
        if self._score_function is not None:
            score = self._score_function()
            priority = score
            if self._score_mode == "min":
                priority = -priority

        # Check if new checkpoint should be accepted
        if len(self._saved) == self._max_saved:
            # Remove old lowest priority checkpoint
            min_index = min(
                range(len(self._saved)), key=lambda i: self._saved[i].priority
            )
            min_priority = self._saved[min_index].priority

            if math.isnan(priority) or priority < min_priority:
                return

            os.remove(self._saved[min_index].filename)
            self._saved.pop(min_index)

        # Get output filename
        step = self._global_step_function()
        filename = self._get_filename(step, score)

        # Save checkpoint
        self._save_checkpoint(filename)

        # Record new checkpoint
        self._saved.append(SavedCheckpoint(filename, priority))

    @staticmethod
    def load_checkpoint(
        accelerator: Accelerator,
        path: str | os.PathLike,
        to_load: Mapping[str, Any],
        to_unwrap: Optional[Sequence[str]] = None,
    ) -> None:
        if to_unwrap is None:
            to_unwrap = []

        ckpt = torch.load(
            f=path,
            map_location="cpu",
            weights_only=True,
        )

        for key, obj in to_load.items():
            if key in to_unwrap:
                obj = accelerator.unwrap_model(obj)
                if is_compiled_module(obj):
                    obj = obj._orig_mod

            obj.load_state_dict(ckpt[key])

    def state_dict(self) -> Dict[str, Any]:
        saved = list(map(tuple, self._saved))
        return dict(saved=saved)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._saved = [SavedCheckpoint(*e) for e in state_dict["saved"]]
