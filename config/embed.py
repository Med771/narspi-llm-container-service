import torch

from transformers import AutoModel


class EmbedConfig:
    torch.set_num_threads(4)

    MODEL_ID = "jinaai/jina-embeddings-v3"
    DEFAULT_DIMS = 512
    BATCH_SIZE = 8
    ROUND_DECIMALS = 6
    MAX_INPUT_CHARS = 32_000

    device = torch.device("cpu")
    model = AutoModel.from_pretrained(MODEL_ID, revision="f1944de", trust_remote_code=True).eval().to(device)
