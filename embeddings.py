
# Produces embeddings for text documents using sentence-transformers.
# Returns numpy arrays suitable for Chroma upsert.

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def encode(self, texts):
        # normalize embeddings -> approximate cosine in inner product space
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")
