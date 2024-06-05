from pydantic import BaseModel
from kornia.augmentation import Normalize
import torch
from torchvision.io import read_image, ImageReadMode, write_jpeg
from torchvision.transforms.functional import convert_image_dtype
from frogbox.service import BaseService


class Request(BaseModel):
    input_path: str
    output_path: str
    quality: int = 95


class Response(BaseModel):
    output_path: str


class SRService(BaseService):
    def inference(self, request: Request):
        model = self.models["sr"]
        config = self.configs["sr"]

        ds_conf = config.datasets["train"].params
        normalize = Normalize(
            mean=ds_conf["normalize_mean"],
            std=ds_conf["normalize_std"],
            keepdim=True,
        )

        image = read_image(request.input_path, ImageReadMode.RGB) / 255.0
        image = normalize(image)

        with torch.inference_mode():
            pred = model(image[None].to(self.device))
            pred = pred.cpu()[0].clamp(0.0, 1.0)

        output = convert_image_dtype(pred, torch.uint8)
        write_jpeg(output, request.output_path, quality=request.quality)
        return Response(output_path=request.output_path)


app = SRService(Request, Response)
