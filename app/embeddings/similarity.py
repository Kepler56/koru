"""
Similarity Matcher Module

Computes semantic similarity between resumes and job descriptions
using cosine similarity on embedding vectors, plus comprehensive
component-based matching for skills, experience, education, projects,
and certifications.
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
    # Enhanced matching components
    component_scores: Dict[str, float] = field(default_factory=dict)
    skill_analysis: Dict[str, Any] = field(default_factory=dict)
    experience_analysis: Dict[str, Any] = field(default_factory=dict)
    education_analysis: Dict[str, Any] = field(default_factory=dict)
    certification_analysis: Dict[str, Any] = field(default_factory=dict)
    project_analysis: Dict[str, Any] = field(default_factory=dict)
    
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
            "component_scores": self.component_scores,
            "skill_analysis": self.skill_analysis,
            "experience_analysis": self.experience_analysis,
            "education_analysis": self.education_analysis,
            "certification_analysis": self.certification_analysis,
            "project_analysis": self.project_analysis,
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


def compute_certification_match(
    resume_certifications: List,
    required_certifications: List[str],
    preferred_certifications: List[str] = None
) -> Dict[str, Any]:
    """
    Compute certification match between resume and job requirements.
    
    Args:
        resume_certifications: List of certifications from resume (can be dicts or strings)
        required_certifications: List of required certification names from job
        preferred_certifications: List of preferred certification names from job
        
    Returns:
        Dictionary with certification match analysis
    """
    # Normalize resume certifications to names
    resume_cert_names = []
    for cert in resume_certifications:
        if isinstance(cert, dict):
            resume_cert_names.append(cert.get('name', '').lower().strip())
        else:
            resume_cert_names.append(str(cert).lower().strip())
    
    resume_cert_set = set(resume_cert_names)
    required_set = {c.lower().strip() for c in required_certifications}
    preferred_set = {c.lower().strip() for c in (preferred_certifications or [])}
    
    # Find matches (partial matching for certifications)
    matched_required = []
    missing_required = []
    
    for req_cert in required_set:
        matched = False
        for res_cert in resume_cert_set:
            # Partial matching - check if key terms match
            req_terms = set(req_cert.split())
            res_terms = set(res_cert.split())
            if len(req_terms & res_terms) >= min(2, len(req_terms)):
                matched_required.append(req_cert)
                matched = True
                break
        if not matched:
            missing_required.append(req_cert)
    
    matched_preferred = []
    for pref_cert in preferred_set:
        for res_cert in resume_cert_set:
            pref_terms = set(pref_cert.split())
            res_terms = set(res_cert.split())
            if len(pref_terms & res_terms) >= min(2, len(pref_terms)):
                matched_preferred.append(pref_cert)
                break
    
    # Calculate scores
    required_coverage = len(matched_required) / len(required_set) if required_set else 1.0
    preferred_coverage = len(matched_preferred) / len(preferred_set) if preferred_set else 0.0
    
    return {
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_preferred": matched_preferred,
        "required_coverage_percentage": required_coverage * 100,
        "preferred_coverage_percentage": preferred_coverage * 100,
        "has_all_required": len(missing_required) == 0,
        "score": required_coverage * 0.8 + preferred_coverage * 0.2,  # Weighted score
    }


def compute_education_match(
    resume_education: List[Dict[str, str]],
    min_education_level: str,
    education_requirements: List[str] = None
) -> Dict[str, Any]:
    """
    Compute education match between resume and job requirements.
    
    Args:
        resume_education: List of education entries from resume
        min_education_level: Minimum education level required (e.g., "Bachelor's", "Master's")
        education_requirements: List of specific education requirements
        
    Returns:
        Dictionary with education match analysis
    """
    # Education level hierarchy
    education_levels = {
        "high school": 1,
        "associate": 2,
        "associate's": 2,
        "bachelor": 3,
        "bachelor's": 3,
        "b.s.": 3,
        "b.a.": 3,
        "master": 4,
        "master's": 4,
        "m.s.": 4,
        "m.a.": 4,
        "mba": 4,
        "phd": 5,
        "ph.d.": 5,
        "doctorate": 5,
        "doctor": 5,
    }
    
    # Get candidate's highest education level
    candidate_level = 0
    candidate_degree = ""
    candidate_field = ""
    candidate_institution = ""
    
    for edu in resume_education:
        degree = edu.get('degree', '').lower()
        for level_name, level_value in education_levels.items():
            if level_name in degree:
                if level_value > candidate_level:
                    candidate_level = level_value
                    candidate_degree = edu.get('degree', '')
                    candidate_field = edu.get('field', '')
                    candidate_institution = edu.get('institution', '')
                break
    
    # Get required level
    required_level = 0
    min_edu_lower = min_education_level.lower() if min_education_level else ""
    for level_name, level_value in education_levels.items():
        if level_name in min_edu_lower:
            required_level = level_value
            break
    
    # Calculate match
    meets_requirement = candidate_level >= required_level if required_level > 0 else True
    level_gap = candidate_level - required_level
    
    # Check field match if specific requirements exist
    field_match = False
    if education_requirements:
        for req in education_requirements:
            req_lower = req.lower()
            if candidate_field and candidate_field.lower() in req_lower:
                field_match = True
                break
            # Check for common field keywords
            field_keywords = ["computer science", "engineering", "mathematics", "data science", 
                           "statistics", "physics", "business", "economics"]
            for keyword in field_keywords:
                if keyword in req_lower and keyword in candidate_field.lower():
                    field_match = True
                    break
    else:
        field_match = True  # No specific field requirement
    
    # Calculate score
    if not meets_requirement:
        score = 0.5  # Below requirement but still considered
    elif level_gap == 0:
        score = 0.8 if field_match else 0.7
    elif level_gap > 0:
        score = 1.0 if field_match else 0.9  # Exceeds requirement
    else:
        score = 0.6
    
    return {
        "meets_requirement": meets_requirement,
        "candidate_level": candidate_degree,
        "candidate_field": candidate_field,
        "candidate_institution": candidate_institution,
        "required_level": min_education_level,
        "level_gap": level_gap,
        "field_match": field_match,
        "score": score,
        "analysis": "Exceeds requirements" if level_gap > 0 else ("Meets requirements" if meets_requirement else "Below requirements"),
    }


def compute_experience_match(
    resume_experience: List[Dict[str, str]],
    total_years_experience: float,
    min_years_required: int,
    max_years_required: int = 0,
    job_title: str = ""
) -> Dict[str, Any]:
    """
    Compute experience match between resume and job requirements.
    
    Args:
        resume_experience: List of experience entries from resume
        total_years_experience: Total calculated years of experience
        min_years_required: Minimum years of experience required
        max_years_required: Maximum years (for seniority matching)
        job_title: Job title for relevance matching
        
    Returns:
        Dictionary with experience match analysis
    """
    # Check years match
    meets_min_years = total_years_experience >= min_years_required if min_years_required > 0 else True
    
    # Check if overqualified (if max is specified)
    overqualified = False
    if max_years_required > 0 and total_years_experience > max_years_required + 3:
        overqualified = True
    
    # Calculate relevance of experience to job title
    relevant_experience_count = 0
    job_title_terms = set(job_title.lower().split()) if job_title else set()
    
    # Remove common words
    common_words = {'senior', 'junior', 'lead', 'staff', 'principal', 'associate', 'the', 'a', 'an'}
    job_title_terms -= common_words
    
    for exp in resume_experience:
        exp_title = exp.get('title', '').lower()
        exp_title_terms = set(exp_title.split()) - common_words
        
        # Check for term overlap
        if job_title_terms and exp_title_terms:
            overlap = len(job_title_terms & exp_title_terms)
            if overlap >= 1 or any(term in exp_title for term in job_title_terms):
                relevant_experience_count += 1
    
    relevance_ratio = relevant_experience_count / len(resume_experience) if resume_experience else 0
    
    # Calculate years gap
    years_gap = total_years_experience - min_years_required
    
    # Calculate score
    if not meets_min_years:
        # Still give partial credit
        score = 0.3 + (total_years_experience / min_years_required) * 0.3 if min_years_required > 0 else 0.5
    elif overqualified:
        score = 0.7  # Overqualified might not be ideal fit
    else:
        base_score = 0.7
        # Bonus for relevant experience
        base_score += relevance_ratio * 0.2
        # Bonus for exceeding requirements (up to 3 years)
        if years_gap > 0:
            base_score += min(years_gap / 3, 1) * 0.1
        score = min(base_score, 1.0)
    
    return {
        "meets_min_years": meets_min_years,
        "total_years": total_years_experience,
        "required_years": f"{min_years_required}-{max_years_required}" if max_years_required else f"{min_years_required}+",
        "years_gap": years_gap,
        "overqualified": overqualified,
        "relevant_experience_count": relevant_experience_count,
        "total_experience_count": len(resume_experience),
        "relevance_ratio": relevance_ratio,
        "score": score,
        "analysis": "Overqualified" if overqualified else ("Meets requirements" if meets_min_years else "Below requirements"),
    }


def compute_project_relevance(
    resume_projects: List[Dict[str, str]],
    job_tech_stack: List[str],
    job_project_types: List[str],
    job_required_skills: List[str]
) -> Dict[str, Any]:
    """
    Compute relevance of candidate's projects to job requirements.
    
    Args:
        resume_projects: List of project entries from resume
        job_tech_stack: Technologies used at the company
        job_project_types: Types of projects mentioned in job
        job_required_skills: Required skills from job
        
    Returns:
        Dictionary with project relevance analysis
    """
    if not resume_projects:
        return {
            "has_projects": False,
            "relevant_projects": [],
            "tech_overlap_score": 0,
            "project_type_match": False,
            "score": 0.5,  # Neutral score if no projects
            "analysis": "No projects provided",
        }
    
    # Normalize job requirements
    job_tech_lower = {t.lower() for t in job_tech_stack}
    job_skills_lower = {s.lower() for s in job_required_skills}
    all_job_techs = job_tech_lower | job_skills_lower
    job_types_lower = {pt.lower() for pt in job_project_types}
    
    relevant_projects = []
    total_tech_matches = 0
    type_matches = 0
    
    for project in resume_projects:
        project_name = project.get('name', '')
        project_desc = project.get('description', '').lower()
        project_techs = [t.lower() for t in project.get('technologies', [])]
        
        # Check tech overlap
        project_tech_set = set(project_techs)
        tech_overlap = project_tech_set & all_job_techs
        
        # Also check description for tech mentions
        for tech in all_job_techs:
            if tech in project_desc:
                tech_overlap.add(tech)
        
        # Check project type relevance
        is_type_match = False
        for job_type in job_types_lower:
            type_keywords = job_type.lower().split()
            for keyword in type_keywords:
                if keyword in project_desc or keyword in project_name.lower():
                    is_type_match = True
                    break
        
        if tech_overlap or is_type_match:
            relevant_projects.append({
                "name": project_name,
                "matching_techs": list(tech_overlap),
                "type_relevant": is_type_match,
            })
            total_tech_matches += len(tech_overlap)
            if is_type_match:
                type_matches += 1
    
    # Calculate scores
    tech_overlap_score = min(total_tech_matches / max(len(all_job_techs), 1), 1.0)
    type_match_ratio = type_matches / len(resume_projects) if resume_projects else 0
    relevance_ratio = len(relevant_projects) / len(resume_projects) if resume_projects else 0
    
    # Combined score
    score = (tech_overlap_score * 0.5 + relevance_ratio * 0.3 + type_match_ratio * 0.2)
    
    return {
        "has_projects": True,
        "total_projects": len(resume_projects),
        "relevant_projects": relevant_projects,
        "relevant_count": len(relevant_projects),
        "tech_overlap_score": tech_overlap_score,
        "type_match_ratio": type_match_ratio,
        "score": score,
        "analysis": f"{len(relevant_projects)} of {len(resume_projects)} projects are relevant",
    }


def compute_comprehensive_match(
    resume,
    job,
    semantic_similarity: float,
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Compute a comprehensive match score combining semantic similarity
    with component-based matching.
    
    SCORING WEIGHTS (must sum to 1.0):
        - semantic: 25% - Text embedding similarity between resume and job
        - skills: 30% - Percentage of required skills the candidate has (HIGHEST WEIGHT)
        - experience: 20% - Years of experience vs. job requirements
        - education: 10% - Degree level and field alignment
        - certifications: 10% - Required and preferred certifications match
        - projects: 5% - Relevance of candidate's projects to job tech stack
    
    FIT LEVELS:
        - Excellent: >= 80% overall score
        - Good: 65-79%
        - Moderate: 50-64%
        - Limited: 35-49%
        - Poor: < 35%
    
    Args:
        resume: ParsedResume object or dict
        job: ParsedJob object or dict
        semantic_similarity: Semantic similarity score (0-1)
        weights: Optional custom weights for components
            
    Returns:
        Dictionary with comprehensive match analysis including:
        - overall_score: Weighted combination of all components
        - fit_level: Human-readable fit assessment
        - component_scores: Individual scores for each component
        - Detailed analysis for skills, experience, education, certs, projects
    """
    # Default weights - skills weighted highest as most important factor
    default_weights = {
        "semantic": 0.25,       # Text similarity
        "skills": 0.30,         # Skills match (HIGHEST - most important)
        "experience": 0.20,     # Experience level
        "education": 0.10,      # Education fit
        "certifications": 0.10, # Certifications
        "projects": 0.05,       # Project relevance
    }
    weights = weights or default_weights
    
    # Get data from objects
    resume_data = resume.to_dict() if hasattr(resume, 'to_dict') else resume
    job_data = job.to_dict() if hasattr(job, 'to_dict') else job
    
    # Compute component scores
    skill_analysis = compute_skill_overlap(
        resume_data.get('skills', []),
        job_data.get('required_skills', [])
    )
    
    certification_analysis = compute_certification_match(
        resume_data.get('certifications', []),
        job_data.get('required_certifications', []),
        job_data.get('preferred_certifications', [])
    )
    
    education_analysis = compute_education_match(
        resume_data.get('education', []),
        job_data.get('min_education_level', ''),
        job_data.get('education_requirements', [])
    )
    
    experience_analysis = compute_experience_match(
        resume_data.get('experience', []),
        resume_data.get('total_years_experience', 0),
        job_data.get('min_years_experience', 0),
        job_data.get('max_years_experience', 0),
        job_data.get('title', '')
    )
    
    project_analysis = compute_project_relevance(
        resume_data.get('projects', []),
        job_data.get('tech_stack', []),
        job_data.get('project_types', []),
        job_data.get('required_skills', [])
    )
    
    # Calculate component scores
    component_scores = {
        "semantic": semantic_similarity,
        "skills": skill_analysis.get('coverage_percentage', 0) / 100,
        "experience": experience_analysis.get('score', 0),
        "education": education_analysis.get('score', 0),
        "certifications": certification_analysis.get('score', 0),
        "projects": project_analysis.get('score', 0),
    }
    
    # Calculate weighted overall score
    overall_score = sum(
        component_scores.get(component, 0) * weight 
        for component, weight in weights.items()
    )
    
    # Determine overall fit level
    if overall_score >= 0.8:
        fit_level = "Excellent"
    elif overall_score >= 0.65:
        fit_level = "Good"
    elif overall_score >= 0.5:
        fit_level = "Moderate"
    elif overall_score >= 0.35:
        fit_level = "Limited"
    else:
        fit_level = "Poor"
    
    return {
        "overall_score": overall_score,
        "fit_level": fit_level,
        "component_scores": component_scores,
        "weights_used": weights,
        "skill_analysis": skill_analysis,
        "certification_analysis": certification_analysis,
        "education_analysis": education_analysis,
        "experience_analysis": experience_analysis,
        "project_analysis": project_analysis,
    }
