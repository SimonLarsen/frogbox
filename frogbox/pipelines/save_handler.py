from typing import Union, Optional, Callable, Mapping, Sequence
import os
from pathlib import Path
import tempfile
import stat
from accelerate import Accelerator
from ignite.handlers.checkpoint import BaseSaveHandler


class NoneSaveHandler(BaseSaveHandler):
    """@private"""

    def __call__(
        self,
        checkpoint: Mapping,
        filename: str,
        metadata: Optional[Mapping] = None,
    ) -> None:
        pass

    def remove(self, filename: str) -> None:
        pass


class AccelerateDiskSaver(BaseSaveHandler):
    """@private"""

    def __init__(
        self,
        dirname: Union[str, os.PathLike],
        accelerator: Accelerator,
        to_unwrap: Optional[Sequence[str]] = None,
        atomic: bool = True,
        **kwargs,
    ):
        self.dirname = Path(dirname).expanduser()
        self.accelerator = accelerator
        self.to_unwrap = to_unwrap
        self.atomic = atomic
        self.kwargs = kwargs

        if not self.dirname.exists():
            self.dirname.mkdir(parents=True)

    def __call__(
        self,
        checkpoint: Mapping,
        filename: str,
        metadata: Optional[Mapping] = None,
    ) -> None:
        to_unwrap = self.to_unwrap if self.to_unwrap else []
        unwrapped_checkpoint = {}
        for key in checkpoint:
            unwrapped_checkpoint[key] = (
                self.accelerator.unwrap_model(checkpoint[key])
                if key in to_unwrap
                else checkpoint[key]
            )

        path = self.dirname / filename
        self._save_func(unwrapped_checkpoint, path, self.accelerator.save)

    def _save_func(
        self, checkpoint: Mapping, path: Path, func: Callable
    ) -> None:
        if not self.atomic:
            func(checkpoint, path, **self.kwargs)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)
            tmp_file = tmp.file
            tmp_name = tmp.name
            try:
                func(checkpoint, tmp_file, **self.kwargs)
            except BaseException:
                tmp.close()
                os.remove(tmp_name)
                raise
            else:
                tmp.close()
                os.replace(tmp.name, path)
                # append group/others read mode
                os.chmod(
                    path, os.stat(path).st_mode | stat.S_IRGRP | stat.S_IROTH
                )

    def remove(self, filename: str) -> None:
        path = self.dirname / filename
        path.unlink()
