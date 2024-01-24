from pydantic import BaseModel
from stort.service import BaseService


class Request(BaseModel):
    pass


class Response(BaseModel):
    pass


class ExampleService(BaseService):
    def inference(self, request: Request):
        return Response()


app = ExampleService(Request, Response)
