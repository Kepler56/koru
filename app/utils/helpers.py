"""
Helper Utilities

Common helper functions used across the application.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dictionary containing all configuration settings
    """
    load_dotenv()
    
    return {
        # Embedding configuration
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
        "sentence_transformer_model": os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        
        # LLM configuration
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "ollama_llm_model": os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        "google_model": os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
        
        # Application settings
        "top_k_matches": int(os.getenv("TOP_K_MATCHES", "5")),
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
    }


def format_score(score: float) -> str:
    """
    Format similarity score as a percentage string.
    
    Args:
        score: Float between 0 and 1
        
    Returns:
        Formatted percentage string (e.g., "85.2%")
    """
    return f"{score * 100:.1f}%"


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = [' '.join(line.split()) for line in lines]
    
    # Remove empty lines but preserve paragraph structure
    result = []
    prev_empty = False
    for line in cleaned_lines:
        if line:
            result.append(line)
            prev_empty = False
        elif not prev_empty:
            result.append("")
            prev_empty = True
    
    return '\n'.join(result).strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length, preserving word boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum character length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length].rsplit(' ', 1)[0]
    return truncated + "..."
