import json
from os import truncate

import torch
import os
import onnxruntime as ort

from transformers import AutoModel, AutoTokenizer, PretrainedConfig

os.makedirs("./models", exist_ok=True)


class EmbedConfig:
    # torch.set_num_threads(4)
    # torch.set_flush_denormal(True)

    # MODEL_ID = "jinaai/jina-embeddings-v3"
    MODEL_ID = "./jina-embeddings-v3"
    ONNX_PATH = os.path.join(MODEL_ID, "onnx", "model.onnx")

    DEFAULT_DIMS = 512
    BATCH_SIZE = 8
    ROUND_DECIMALS = 6
    MAX_INPUT_CHARS = 32_000

    # оптимизация ONNX
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    session_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session_options.add_session_config_entry("session.inter_op.allow_spinning", "1")

    config_path = os.path.join(MODEL_ID, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    lora_adaptations = config_data.get("lora_adaptations", [
        "retrieval.passage",
        "retrieval.query",
    ])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                              cache_dir="/models",
                                              trust_remote_code=True)

    session = ort.InferenceSession(
        ONNX_PATH,
        providers=['CPUExecutionProvider'],
        sess_options=session_options,
    )

    config = type('Config', (), {})()
    config.lora_adaptations = lora_adaptations

    # device = torch.device("cpu")
    # model = AutoModel.from_pretrained(MODEL_ID,
    #                                   revision="f1944de",
    #                                   trust_remote_code=True,
    #                                   ).eval().to(device)
