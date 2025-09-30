import torch
import numpy as np

from typing import List

from config import EmbedConfig


class EmbedHelper:
    @staticmethod
    @torch.inference_mode()
    def embed_docs(texts: List[str],
                   dims: int = EmbedConfig.DEFAULT_DIMS,
                   batch_size: int = EmbedConfig.BATCH_SIZE) -> List[List[float]]:
        results: List[List[float]] = []

        for start in range(0, len(texts), batch_size):
            safe_batch = []

            for t in texts[start:start + batch_size]:
                if not isinstance(t, str): continue

                safe_batch.append(t)

            if not safe_batch: continue

            embs = EmbedConfig.model.encode(
                safe_batch,
                task="retrieval.passage",
                normalize_embeddings=True,
                truncate_dim=dims
            )

            for e in embs:
                if isinstance(e, torch.Tensor):
                    arr = e.detach().cpu().numpy()
                else:
                    arr = np.asarray(e)

                arr = np.round(arr.astype(np.float32), EmbedConfig.ROUND_DECIMALS)

                results.append(arr.tolist())

        return results

    @staticmethod
    @torch.inference_mode()
    def embed_query(query: str,
                    dims: int = EmbedConfig.DEFAULT_DIMS) -> List[float]:
        try:
            embs = EmbedConfig.model.encode(
                [query],
                task="retrieval.query",
                normalize_embeddings=True,
                truncate_dim=dims
            )
        except Exception:
            return []

        if len(embs) == 1:
            if isinstance(embs[0], torch.Tensor):
                arr = embs[0].detach().cpu().numpy()
            else:
                arr = np.asarray(embs[0])

            arr = np.round(arr.astype(np.float32), EmbedConfig.ROUND_DECIMALS)

            return arr.tolist()

        return []
