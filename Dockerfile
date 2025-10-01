FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision torchaudio transformers

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('jinaai/jina-embeddings-v3', trust_remote_code=True)"

WORKDIR /app
COPY . /app

CMD ["python", "main.py"]
