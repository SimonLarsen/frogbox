from typing import Dict, Type
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import torch
from .config import Config
from .utils import load_model_checkpoint


class BaseServiceSettings(BaseSettings):
    checkpoints: Dict[str, str] = {}
    device: str = "cpu"


class HealthResponse(BaseModel):
    status: str = "OK"


class BaseService(FastAPI):
    configs: Dict[str, Config]
    models: Dict[str, torch.nn.Module]
    device: torch.device

    def __init__(
        self,
        request_class: Type[BaseModel],
        response_class: Type[BaseModel],
    ):
        self.settings = BaseServiceSettings()

        super().__init__(lifespan=self._lifespan)

        @self.post("/inference")
        async def do_inference(
            request: request_class,  # type: ignore
        ) -> response_class:  # type: ignore
            """Run model inference."""
            if not hasattr(self, "inference"):
                raise RuntimeError("inference method not implemented.")
            return self.inference(request)

        @self.get("/health")
        async def get_health() -> HealthResponse:
            """Get service health."""
            return HealthResponse(status="OK")

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self.on_startup()
        yield
        self.on_shutdown()

    def on_startup(self):
        self.configs = {}
        self.models = {}
        self.device = torch.device(self.settings.device)

        for name, path in self.settings.checkpoints.items():
            model, config = load_model_checkpoint(path)

            model = model.eval().to(self.device)
            self.models[name] = model
            self.configs[name] = config

    def on_shutdown(self):
        del self.models
        torch.cuda.empty_cache()
