from helper.embed import EmbedHelper


class RefactorHelper:
    @staticmethod
    def get_query_embedding(obj: dict):
        res = {
            "uuid": obj['uuid'],
            "embedding": []
        }

        try:
            embedding: list[float] = EmbedHelper.embed_query(query=obj["query"])
        except Exception:
            return res

        if not embedding: return res

        res["embedding"] = embedding

        return res

    @staticmethod
    def get_docs_embedding(obj: dict):
        res = {
            "uuid": obj['uuid'],
            "embeddings": []
        }

        try:
            embeddings: list[list[float]] = EmbedHelper.embed_docs(texts=obj["chunks"])
        except Exception:
            return res

        if not embeddings: return res

        res["embeddings"] = embeddings

        return res
