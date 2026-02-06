from helper.embed import EmbedHelper
from config import EmbedConfig


class RefactorHelper:
    @staticmethod
    def get_query_embedding(obj: dict):
        res = {
            "uuid": obj['uuid'],
            "embedding": [],
            "error": None
        }

        # Проверка длины запроса
        if len(obj["query"]) > EmbedConfig.MAX_INPUT_CHARS:
            res["error"] = f"Query too long: {len(obj['query'])} chars, max allowed: {EmbedConfig.MAX_INPUT_CHARS}"
            return res

        try:
            embedding: list[float] = EmbedHelper.embed_query(query=obj["query"])
        except Exception as e:
            res["error"] = f"Embedding error: {str(e)}"
            return res

        if not embedding:
            res["error"] = "Empty embedding result"
            return res

        res["embedding"] = embedding

        return res

    @staticmethod
    def get_docs_embedding(obj: dict):
        res = {
            "uuid": obj['uuid'],
            "embeddings": [],
            "errors": []
        }

        # Фильтруем чанки по длине
        valid_chunks = []
        invalid_chunks_indices = []

        for i, chunk in enumerate(obj["chunks"]):
            if len(chunk) > EmbedConfig.MAX_INPUT_CHARS:
                invalid_chunks_indices.append({
                    "index": i,
                    "length": len(chunk),
                    "max_allowed": EmbedConfig.MAX_INPUT_CHARS,
                    "error": "Chunk too long"
                })
            else:
                valid_chunks.append(chunk)

        # Добавляем информацию об ошибках
        if invalid_chunks_indices:
            res["errors"] = invalid_chunks_indices

        try:
            embeddings: list[list[float]] = EmbedHelper.embed_docs(texts=valid_chunks)
        except Exception as e:
            res["error"] = f"Embedding error: {str(e)}"
            return res

        if not embeddings:
            res["error"] = "Empty embeddings result"
            return res

        res["embeddings"] = embeddings

        return res
