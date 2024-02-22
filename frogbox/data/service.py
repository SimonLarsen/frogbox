from pydantic import BaseModel
from frogbox.service import BaseService


class Request(BaseModel):
    pass


class Response(BaseModel):
    pass


class ExampleService(BaseService):
    def inference(self, request: Request):
        return Response()


app = ExampleService(Request, Response)
