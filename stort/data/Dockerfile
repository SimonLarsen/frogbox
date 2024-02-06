ARG PYTORCH_VERSION=2.1.2-cuda11.8-cudnn8-runtime
ARG PYTORCH_PLATFORM=linux/amd64
ARG CHECKPOINT

FROM --platform=${PYTORCH_PLATFORM} pytorch/pytorch:${PYTORCH_VERSION}

EXPOSE 8000

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY service.py ./
COPY {{ requirements }} ./requirements.txt
COPY models/* ./models/

RUN pip install -r requirements.txt
RUN pip install -U uvicorn

{% for ckpt in checkpoints %}
RUN mkdir -p {{ ckpt.parent_path }}
COPY {{ ckpt.model_path }} {{ ckpt.model_path }}
COPY {{ ckpt.config_path }} {{ ckpt.config_path }}
{% endfor %}

ENV CHECKPOINTS='{{ env_checkpoints }}'

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]