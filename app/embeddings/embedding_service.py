"""
Embedding Service Module

Provides a unified interface for generating text embeddings using
SentenceTransformers, Ollama, or Google Vertex AI.
"""

import os
from typing import List, Union, Optional
from abc import ABC, abstractmethod
import numpy as np

from dotenv import load_dotenv

load_dotenv()


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            NumPy array of embeddings (n_texts x embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        pass


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Embedding provider using SentenceTransformers.
    
    Default model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
    Alternative: all-mpnet-base-v2 (higher quality, 768 dimensions)
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize SentenceTransformer provider.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name or os.getenv(
            "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
        )
        print(f"Loading SentenceTransformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self._embedding_dim}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using SentenceTransformer."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider using Ollama's local embedding models.
    
    Default model: nomic-embed-text (768 dimensions)
    Alternative: mxbai-embed-large (1024 dimensions)
    """
    
    def __init__(self, model_name: str = None, base_url: str = None):
        """
        Initialize Ollama embedding provider.
        
        Args:
            model_name: Name of the Ollama embedding model
            base_url: Ollama server URL
        """
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama is required. Install with: pip install ollama"
            )
        
        self.model_name = model_name or os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
        )
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        
        self._client = ollama.Client(host=self.base_url)
        self._embedding_dim = None
        
        print(f"Using Ollama embeddings: {self.model_name} at {self.base_url}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Ollama."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            response = self._client.embeddings(
                model=self.model_name,
                prompt=text
            )
            embedding = response["embedding"]
            embeddings.append(embedding)
            
            if self._embedding_dim is None:
                self._embedding_dim = len(embedding)
        
        return np.array(embeddings)
    
    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            # Get dimension by embedding a test string
            self.embed("test")
        return self._embedding_dim


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider using Google's Generative AI embeddings.
    
    Default model: models/embedding-001 (768 dimensions)
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Google embedding provider.
        
        Args:
            api_key: Google API key
            model_name: Embedding model name
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required. Install with: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self._genai = genai
        
        self.model_name = model_name or "models/embedding-001"
        self._embedding_dim = 768  # Default for embedding-001
        
        print(f"Using Google embeddings: {self.model_name}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Google Generative AI."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            result = self._genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        
        return np.array(embeddings)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class EmbeddingService:
    """
    Unified embedding service that supports multiple providers.
    
    Providers:
        - sentence-transformers: Local, fast, good quality
        - ollama: Local Ollama server
        - google: Google Generative AI API
    
    Example:
        >>> service = EmbeddingService(provider="sentence-transformers")
        >>> embeddings = service.embed(["Hello world", "How are you"])
        >>> print(embeddings.shape)
        (2, 384)
    """
    
    PROVIDERS = {
        "sentence-transformers": SentenceTransformerProvider,
        "ollama": OllamaEmbeddingProvider,
        "google": GoogleEmbeddingProvider,
    }
    
    def __init__(self, provider: str = None, **kwargs):
        """
        Initialize the embedding service.
        
        Args:
            provider: Provider name ("sentence-transformers", "ollama", "google")
            **kwargs: Additional arguments passed to the provider
        """
        self.provider_name = provider or os.getenv(
            "EMBEDDING_PROVIDER", "sentence-transformers"
        )
        
        if self.provider_name not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {self.provider_name}. "
                f"Available: {list(self.PROVIDERS.keys())}"
            )
        
        provider_class = self.PROVIDERS[self.provider_name]
        self._provider = provider_class(**kwargs)
        
        # Cache for storing embeddings
        self._cache = {}
    
    def embed(self, texts: Union[str, List[str]], use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Single text or list of texts to embed
            use_cache: Whether to use/store cached embeddings
            
        Returns:
            NumPy array of embeddings (n_texts x embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if use_cache:
            # Check cache for existing embeddings
            results = []
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(texts):
                cache_key = hash(text)
                if cache_key in self._cache:
                    results.append((i, self._cache[cache_key]))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
            
            # Embed uncached texts
            if texts_to_embed:
                new_embeddings = self._provider.embed(texts_to_embed)
                
                # Store in cache and results
                for idx, (orig_idx, text) in enumerate(zip(text_indices, texts_to_embed)):
                    embedding = new_embeddings[idx]
                    cache_key = hash(text)
                    self._cache[cache_key] = embedding
                    results.append((orig_idx, embedding))
            
            # Sort by original index and return
            results.sort(key=lambda x: x[0])
            return np.array([r[1] for r in results])
        else:
            return self._provider.embed(texts)
    
    def embed_resume(self, resume) -> np.ndarray:
        """
        Generate embedding for a parsed resume.
        
        Args:
            resume: ParsedResume object
            
        Returns:
            Embedding vector
        """
        text = resume.to_text_for_embedding()
        return self.embed(text)[0]
    
    def embed_job(self, job) -> np.ndarray:
        """
        Generate embedding for a parsed job description.
        
        Args:
            job: ParsedJob object
            
        Returns:
            Embedding vector
        """
        text = job.to_text_for_embedding()
        return self.embed(text)[0]
    
    def embed_documents(self, documents: List) -> np.ndarray:
        """
        Generate embeddings for multiple documents (resumes or jobs).
        
        Args:
            documents: List of ParsedResume or ParsedJob objects
            
        Returns:
            Array of embeddings (n_documents x embedding_dim)
        """
        texts = [doc.to_text_for_embedding() for doc in documents]
        return self.embed(texts)
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        return self._provider.embedding_dim
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
    
    @property
    def provider(self) -> str:
        """Return the current provider name."""
        return self.provider_name
