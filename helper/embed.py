import torch
import numpy as np

from typing import List

from config import EmbedConfig


class EmbedHelper:
    @staticmethod
    def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray):
        token_embeddings = model_output
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    @staticmethod
    def _tokenize(texts):
        return EmbedConfig.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,  # jina v3 поддерживает до 8192 токенов
            return_tensors="np"
        )

    @staticmethod
    # @torch.inference_mode()
    def embed_docs(texts: List[str],
                   dims: int = EmbedConfig.DEFAULT_DIMS,
                   batch_size: int = EmbedConfig.BATCH_SIZE) -> List[List[float]]:
        results: List[List[float]] = []

        texts = [t for t in texts if isinstance(t, str) and t.strip()]

        if not texts: return []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if not batch: continue

            inputs = EmbedHelper._tokenize(batch)

            task_type = 'retrieval.passage'
            task_id = np.array(EmbedConfig.config.lora_adaptations.index(task_type), dtype=np.int64)
            ort_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "task_id": task_id
            }

            raw_output = EmbedConfig.session.run(None, ort_inputs)[0]

            embedding = EmbedHelper.mean_pooling(raw_output, inputs["attention_mask"])

            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embedding = embedding / norms

            embedding = embedding[:, :512]

            embedding = np.round(embedding.astype(np.float32), decimals=EmbedConfig.ROUND_DECIMALS)
            results.extend(embedding.tolist())  # list[list[float]]

        return results
            # embs = EmbedConfig.model.encode(
            #     batch,
            #     task="retrieval.passage",
            #     normalize_embeddings=True,
            #     truncate_dim=dims
            # )
            #
            # Векторизованная обработка эмбеддингов
            # if isinstance(embs, torch.Tensor):
            #     arrays = embs.detach().cpu().numpy()
            # else:
            #     arrays = np.asarray(embs)
            #
            # Векторизованное округление
            # arrays = np.round(arrays.astype(np.float32), EmbedConfig.ROUND_DECIMALS)
            #
            # results.extend(arrays.tolist())

        return results

    @staticmethod
    # @torch.inference_mode()
    def embed_query(query: str,
                    dims: int = EmbedConfig.DEFAULT_DIMS) -> List[float]:

        if not query.strip():
            return []

        inputs = EmbedHelper._tokenize(query.strip())

        task_type = 'retrieval.query'
        task_id = np.array(EmbedConfig.config.lora_adaptations.index(task_type), dtype=np.int64)

        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "task_id": task_id
        }

        outputs = EmbedConfig.session.run(None, ort_inputs)[0]  # [1024]

        embeddings = EmbedHelper.mean_pooling(outputs, inputs["attention_mask"])
        embeddings = embeddings / np.linalg.norm(embeddings , ord=2, axis=1, keepdims=True)

        embs = embeddings[:, :dims]

        # try:
        #     embs = EmbedConfig.model.encode(
        #         [query],
        #         task="retrieval.query",
        #         normalize_embeddings=True,
        #         truncate_dim=dims
        #     )
        # except Exception:
        #     return []
        #
        if len(embs) == 1:
            if isinstance(embs[0], torch.Tensor):
                arr = embs[0].detach().cpu().numpy()
            else:
                arr = np.asarray(embs[0])

            arr = np.round(arr.astype(np.float32), EmbedConfig.ROUND_DECIMALS)

            return arr.tolist()

        return []
