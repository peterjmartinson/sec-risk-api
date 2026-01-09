import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # This model produces 384-dimensional vectors
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> npt.NDArray[np.float32]:
        """
        Turns a list of strings into a NumPy array of embeddings.
        Type annotated for mypy compliance.
        """
        # convert_to_numpy=True is default, but we're being explicit
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def get_similarity(self, text_a: str, text_b: str) -> float:
        """
        Success Condition: Combat test helper to verify semantic proximity.
        """
        vecs = self.encode([text_a, text_b])
        # Cosine Similarity: (A dot B) / (||A|| * ||B||)
        # Sentence-transformers usually returns normalized vectors, 
        # so a simple dot product works, but let's be robust:
        norm_a = vecs[0] / np.linalg.norm(vecs[0])
        norm_b = vecs[1] / np.linalg.norm(vecs[1])
        return float(np.dot(norm_a, norm_b))
