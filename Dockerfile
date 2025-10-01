FROM python:3.13-slim

# Устанавливаем системные пакеты
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('jinaai/jina-embeddings-v3', revision='f1944de', trust_remote_code=True)"

COPY . /app

CMD ["python", "main.py"]
