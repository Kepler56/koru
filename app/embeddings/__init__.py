"""
Embeddings Module

Handles vector embedding generation and semantic similarity computation
using SentenceTransformers, Ollama, or Vertex AI.
"""

from .embedding_service import EmbeddingService
from .similarity import SimilarityMatcher

__all__ = ["EmbeddingService", "SimilarityMatcher"]
