import os
from typing import List, Optional, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from opensearchpy import OpenSearch, helpers

# Initialize embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class VectorStore:
    """
    Abstraction over Qdrant and OpenSearch for storing and querying embeddings.

    Supports upsert, search, and deletion operations across backends.
    """
    def __init__(
        self,
        backend: str = "qdrant",
        qdrant_url: Optional[str] = None,
        opensearch_hosts: Optional[List[Dict[str, Any]]] = None,
        index_name: str = "documents",
        embedding_model: Optional[str] = None,
    ):
        self.backend = backend.lower()
        self.index_name = index_name
        self.model = SentenceTransformer(embedding_model or EMBEDDING_MODEL)

        if self.backend == "qdrant":
            self.client = QdrantClient(url=qdrant_url or os.getenv("QDRANT_URL"),
                                       api_key=os.getenv("QDRANT_API_KEY"))
            # Ensure collection exists
            self.client.recreate_collection(
                collection_name=self.index_name,
                vectors_config=qdrant_models.VectorParams(size=self.model.get_sentence_embedding_dimension(), distance=qdrant_models.Distance.COSINE),
            )
        elif self.backend == "opensearch":
            hosts = opensearch_hosts or [{"host": os.getenv("OPENSEARCH_HOST"), "port": int(os.getenv("OPENSEARCH_PORT", "9200"))}]
            self.client = OpenSearch(hosts=hosts)
            # Create index mapping if not exists
            if not self.client.indices.exists(self.index_name):
                body = {
                    "mappings": {
                        "properties": {
                            "embedding": {"type": "dense_vector", "dims": self.model.get_sentence_embedding_dimension()},
                            "text": {"type": "text"},
                        }
                    }
                }
                self.client.indices.create(index=self.index_name, body=body)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def upsert(self, ids: List[int], texts: List[str]) -> None:
        """
        Encode texts and upsert into the vector store.
        """
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

        if self.backend == "qdrant":
            points = [
                qdrant_models.PointStruct(id=id_, vector=emb, payload={"text": txt})
                for id_, emb, txt in zip(ids, embeddings, texts)
            ]
            self.client.upsert(collection_name=self.index_name, points=points)
        else:  # opensearch
            actions = []
            for id_, emb, txt in zip(ids, embeddings, texts):
                doc = {"embedding": emb, "text": txt}
                actions.append({
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": id_,
                    "_source": doc,
                })
            helpers.bulk(self.client, actions)

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search: encode query and return top_k results above threshold.
        """
        query_emb = self.model.encode([query_text])[0].tolist()

        if self.backend == "qdrant":
            response = self.client.search(
                collection_name=self.index_name,
                query_vector=query_emb,
                limit=top_k,
            )
            return [
                {"id": hit.id, "score": hit.score, "text": hit.payload.get("text")} for hit in response
            ]
        else:
            body = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_emb},
                        },
                    }
                }
            }
            resp = self.client.search(index=self.index_name, body=body)
            results = []
            for hit in resp["hits"]["hits"]:
                score = hit["_score"]
                if score >= score_threshold:
                    results.append({"id": hit["_id"], "score": score, "text": hit["_source"]["text"]})
            return results

    def delete(self, ids: List[int]) -> None:
        """
        Delete documents by IDs from the vector store.
        """
        if self.backend == "qdrant":
            self.client.delete_collection(collection_name=self.index_name, where=qdrant_models.Filter())
        else:
            actions = [
                {"_op_type": "delete", "_index": self.index_name, "_id": id_}
                for id_ in ids
            ]
            helpers.bulk(self.client, actions)
