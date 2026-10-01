from sentence_transformers import SentenceTransformer
import numpy as np
from graphrag.embeddings.embedder import Embedder

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)