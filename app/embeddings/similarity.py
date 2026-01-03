"""
Similarity Matcher Module

Computes semantic similarity between resumes and job descriptions
using cosine similarity on embedding vectors.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MatchResult:
    """Result of matching a resume to a job."""
    resume_index: int
    job_index: int
    similarity_score: float
    resume_name: str = ""
    job_title: str = ""
    resume_data: Dict[str, Any] = field(default_factory=dict)
    job_data: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "resume_index": self.resume_index,
            "job_index": self.job_index,
            "similarity_score": self.similarity_score,
            "similarity_percentage": f"{self.similarity_score * 100:.1f}%",
            "resume_name": self.resume_name,
            "job_title": self.job_title,
            "explanation": self.explanation,
        }


class SimilarityMatcher:
    """
    Computes semantic similarity between documents using their embeddings.
    
    Supports:
        - Cosine similarity (default)
        - Euclidean distance
        - Dot product similarity
    
    Example:
        >>> matcher = SimilarityMatcher()
        >>> resume_embeddings = embedding_service.embed_documents(resumes)
        >>> job_embeddings = embedding_service.embed_documents(jobs)
        >>> matches = matcher.match_resumes_to_jobs(
        ...     resume_embeddings, job_embeddings, 
        ...     resumes, jobs,
        ...     top_k=5
        ... )
    """
    
    def __init__(self, similarity_metric: str = "cosine"):
        """
        Initialize the similarity matcher.
        
        Args:
            similarity_metric: One of "cosine", "euclidean", "dot"
        """
        self.similarity_metric = similarity_metric
        
        self._similarity_functions = {
            "cosine": self._cosine_similarity,
            "euclidean": self._euclidean_similarity,
            "dot": self._dot_product_similarity,
        }
        
        if similarity_metric not in self._similarity_functions:
            raise ValueError(
                f"Unknown similarity metric: {similarity_metric}. "
                f"Available: {list(self._similarity_functions.keys())}"
            )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of vectors.
        
        Args:
            a: Array of shape (n, dim)
            b: Array of shape (m, dim)
            
        Returns:
            Similarity matrix of shape (n, m)
        """
        # Normalize vectors
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        
        # Compute cosine similarity
        return np.dot(a_norm, b_norm.T)
    
    def _euclidean_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute similarity based on Euclidean distance.
        
        Returns similarity scores (higher = more similar).
        """
        # Compute pairwise distances
        a_sq = np.sum(a ** 2, axis=1, keepdims=True)
        b_sq = np.sum(b ** 2, axis=1, keepdims=True)
        distances = np.sqrt(a_sq + b_sq.T - 2 * np.dot(a, b.T) + 1e-9)
        
        # Convert distance to similarity (1 / (1 + distance))
        return 1.0 / (1.0 + distances)
    
    def _dot_product_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute dot product similarity."""
        return np.dot(a, b.T)
    
    def compute_similarity_matrix(
        self, 
        embeddings_a: np.ndarray, 
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise similarity between two sets of embeddings.
        
        Args:
            embeddings_a: Array of shape (n, dim) - e.g., resume embeddings
            embeddings_b: Array of shape (m, dim) - e.g., job embeddings
            
        Returns:
            Similarity matrix of shape (n, m)
        """
        similarity_fn = self._similarity_functions[self.similarity_metric]
        return similarity_fn(embeddings_a, embeddings_b)
    
    def match_resumes_to_jobs(
        self,
        resume_embeddings: np.ndarray,
        job_embeddings: np.ndarray,
        resumes: List = None,
        jobs: List = None,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[List[MatchResult]]:
        """
        Match each resume to the most similar jobs.
        
        Args:
            resume_embeddings: Array of resume embeddings (n_resumes x dim)
            job_embeddings: Array of job embeddings (n_jobs x dim)
            resumes: Optional list of ParsedResume objects for metadata
            jobs: Optional list of ParsedJob objects for metadata
            top_k: Number of top matches to return per resume
            threshold: Minimum similarity score threshold
            
        Returns:
            List of lists, where each inner list contains top matches for a resume
        """
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(
            resume_embeddings, job_embeddings
        )
        
        all_matches = []
        
        for resume_idx in range(len(resume_embeddings)):
            # Get similarity scores for this resume
            scores = similarity_matrix[resume_idx]
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            resume_matches = []
            for job_idx in top_indices:
                score = scores[job_idx]
                
                # Skip if below threshold
                if score < threshold:
                    continue
                
                # Create match result
                match = MatchResult(
                    resume_index=resume_idx,
                    job_index=int(job_idx),
                    similarity_score=float(score),
                )
                
                # Add metadata if available
                if resumes:
                    resume = resumes[resume_idx]
                    match.resume_name = getattr(resume, 'name', '') or getattr(resume, 'file_name', '')
                    match.resume_data = resume.to_dict() if hasattr(resume, 'to_dict') else {}
                
                if jobs:
                    job = jobs[job_idx]
                    match.job_title = getattr(job, 'title', '') or getattr(job, 'file_name', '')
                    match.job_data = job.to_dict() if hasattr(job, 'to_dict') else {}
                
                resume_matches.append(match)
            
            all_matches.append(resume_matches)
        
        return all_matches
    
    def match_jobs_to_resumes(
        self,
        job_embeddings: np.ndarray,
        resume_embeddings: np.ndarray,
        jobs: List = None,
        resumes: List = None,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[List[MatchResult]]:
        """
        Match each job to the most similar resumes.
        
        Args:
            job_embeddings: Array of job embeddings (n_jobs x dim)
            resume_embeddings: Array of resume embeddings (n_resumes x dim)
            jobs: Optional list of ParsedJob objects for metadata
            resumes: Optional list of ParsedResume objects for metadata
            top_k: Number of top matches to return per job
            threshold: Minimum similarity score threshold
            
        Returns:
            List of lists, where each inner list contains top matches for a job
        """
        # Compute similarity matrix (jobs x resumes)
        similarity_matrix = self.compute_similarity_matrix(
            job_embeddings, resume_embeddings
        )
        
        all_matches = []
        
        for job_idx in range(len(job_embeddings)):
            # Get similarity scores for this job
            scores = similarity_matrix[job_idx]
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            job_matches = []
            for resume_idx in top_indices:
                score = scores[resume_idx]
                
                # Skip if below threshold
                if score < threshold:
                    continue
                
                # Create match result
                match = MatchResult(
                    resume_index=int(resume_idx),
                    job_index=job_idx,
                    similarity_score=float(score),
                )
                
                # Add metadata if available
                if jobs:
                    job = jobs[job_idx]
                    match.job_title = getattr(job, 'title', '') or getattr(job, 'file_name', '')
                    match.job_data = job.to_dict() if hasattr(job, 'to_dict') else {}
                
                if resumes:
                    resume = resumes[resume_idx]
                    match.resume_name = getattr(resume, 'name', '') or getattr(resume, 'file_name', '')
                    match.resume_data = resume.to_dict() if hasattr(resume, 'to_dict') else {}
                
                job_matches.append(match)
            
            all_matches.append(job_matches)
        
        return all_matches
    
    def find_best_match(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidates: List = None
    ) -> Tuple[int, float, Optional[Any]]:
        """
        Find the single best match for a query embedding.
        
        Args:
            query_embedding: Single embedding vector (dim,)
            candidate_embeddings: Array of candidate embeddings (n x dim)
            candidates: Optional list of candidate objects
            
        Returns:
            Tuple of (best_index, similarity_score, candidate_object)
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        similarity_matrix = self.compute_similarity_matrix(
            query_embedding, candidate_embeddings
        )
        
        scores = similarity_matrix[0]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        
        candidate = candidates[best_idx] if candidates else None
        
        return best_idx, best_score, candidate
    
    def rank_all(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidates: List = None
    ) -> List[Tuple[int, float, Optional[Any]]]:
        """
        Rank all candidates by similarity to query.
        
        Args:
            query_embedding: Single embedding vector (dim,)
            candidate_embeddings: Array of candidate embeddings (n x dim)
            candidates: Optional list of candidate objects
            
        Returns:
            List of (index, score, candidate) tuples sorted by score descending
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute similarities
        similarity_matrix = self.compute_similarity_matrix(
            query_embedding, candidate_embeddings
        )
        
        scores = similarity_matrix[0]
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in sorted_indices:
            candidate = candidates[idx] if candidates else None
            results.append((int(idx), float(scores[idx]), candidate))
        
        return results


def compute_skill_overlap(resume_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
    """
    Compute skill overlap between resume and job.
    
    Args:
        resume_skills: List of skills from resume
        job_skills: List of required skills from job
        
    Returns:
        Dictionary with overlap analysis
    """
    # Normalize skills for comparison
    resume_skills_lower = {s.lower().strip() for s in resume_skills}
    job_skills_lower = {s.lower().strip() for s in job_skills}
    
    # Find matches
    matching_skills = resume_skills_lower & job_skills_lower
    missing_skills = job_skills_lower - resume_skills_lower
    extra_skills = resume_skills_lower - job_skills_lower
    
    # Calculate coverage
    coverage = len(matching_skills) / len(job_skills_lower) if job_skills_lower else 0.0
    
    return {
        "matching_skills": list(matching_skills),
        "missing_skills": list(missing_skills),
        "extra_skills": list(extra_skills),
        "coverage_percentage": coverage * 100,
        "matched_count": len(matching_skills),
        "total_required": len(job_skills_lower),
    }
