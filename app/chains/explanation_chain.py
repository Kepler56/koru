"""
Explanation Chain Module

Uses LangChain to generate AI-powered explanations for resume-job matches,
providing human-readable reasoning for why a candidate fits a position.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MatchExplanation:
    """Structured explanation for a resume-job match."""
    summary: str
    strengths: List[str]
    gaps: List[str]
    recommendations: List[str]
    fit_score_reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "strengths": self.strengths,
            "gaps": self.gaps,
            "recommendations": self.recommendations,
            "fit_score_reasoning": self.fit_score_reasoning,
        }
    
    def to_text(self) -> str:
        """Convert to formatted text."""
        parts = [f"**Summary:** {self.summary}"]
        
        if self.strengths:
            parts.append("\n**Strengths:**")
            for s in self.strengths:
                parts.append(f"  • {s}")
        
        if self.gaps:
            parts.append("\n**Gaps:**")
            for g in self.gaps:
                parts.append(f"  • {g}")
        
        if self.recommendations:
            parts.append("\n**Recommendations:**")
            for r in self.recommendations:
                parts.append(f"  • {r}")
        
        if self.fit_score_reasoning:
            parts.append(f"\n**Fit Score Reasoning:** {self.fit_score_reasoning}")
        
        return "\n".join(parts)


class ExplanationChain:
    """
    LangChain-based chain for generating match explanations.
    
    Supports:
        - Ollama (local LLM)
        - Google Generative AI (Gemini)
    
    Example:
        >>> chain = ExplanationChain(provider="ollama")
        >>> explanation = chain.explain_match(resume, job, similarity_score=0.85)
        >>> print(explanation.summary)
    """
    
    EXPLANATION_PROMPT = """You are an expert HR analyst helping to explain why a candidate might be a good fit for a job position.

Given the following resume and job description, provide a detailed analysis of the match.

## Resume Information:
{resume_info}

## Job Description:
{job_info}

## Similarity Score: {similarity_score}%

Please provide your analysis in the following format:

SUMMARY: A 2-3 sentence overview of how well the candidate matches the job.

STRENGTHS:
- List 3-5 specific strengths that make this candidate a good fit
- Focus on matching skills, relevant experience, and qualifications

GAPS:
- List any missing skills or qualifications (if any)
- Be constructive and specific

RECOMMENDATIONS:
- Provide 2-3 actionable recommendations for the candidate
- Or suggestions for the hiring manager to consider

FIT_SCORE_REASONING: Explain why the similarity score makes sense given the match.

Be specific, professional, and constructive in your analysis."""

    def __init__(self, provider: str = None, **kwargs):
        """
        Initialize the explanation chain.
        
        Args:
            provider: LLM provider ("ollama" or "google")
            **kwargs: Additional arguments for the LLM
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self._llm = None
        self._chain = None
        self._kwargs = kwargs
        
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize the LangChain components."""
        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "google":
            self._init_google()
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'ollama' or 'google'.")
    
    def _init_ollama(self):
        """Initialize Ollama LLM."""
        try:
            from langchain_ollama import OllamaLLM as Ollama
        except ImportError:
            try:
                from langchain_community.llms import Ollama
            except ImportError:
                raise ImportError(
                    "langchain-ollama is required. "
                    "Install with: pip install langchain-ollama"
                )
        
        model_name = self._kwargs.get("model") or os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
        base_url = self._kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        self._llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.3,
        )
        
        print(f"Initialized Ollama LLM: {model_name} at {base_url}")
    
    def _init_google(self):
        """Initialize Google Generative AI LLM."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required. "
                "Install with: pip install langchain-google-genai"
            )
        
        api_key = self._kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        model_name = self._kwargs.get("model") or os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
        
        self._llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )
        
        print(f"Initialized Google Generative AI: {model_name}")
    
    def _format_resume_info(self, resume) -> str:
        """Format resume information for the prompt."""
        if hasattr(resume, 'to_dict'):
            data = resume.to_dict()
        elif isinstance(resume, dict):
            data = resume
        else:
            return str(resume)
        
        parts = []
        
        if data.get('name'):
            parts.append(f"Name: {data['name']}")
        
        if data.get('summary'):
            parts.append(f"Summary: {data['summary']}")
        
        if data.get('skills'):
            skills = data['skills']
            if isinstance(skills, list):
                parts.append(f"Skills: {', '.join(skills)}")
            else:
                parts.append(f"Skills: {skills}")
        
        if data.get('experience'):
            parts.append("Experience:")
            for exp in data['experience'][:5]:
                if isinstance(exp, dict):
                    title = exp.get('title', '')
                    company = exp.get('company', '')
                    desc = exp.get('description', '')[:200]
                    parts.append(f"  - {title} at {company}: {desc}")
                else:
                    parts.append(f"  - {exp}")
        
        if data.get('education'):
            parts.append("Education:")
            for edu in data['education'][:3]:
                if isinstance(edu, dict):
                    degree = edu.get('degree', '')
                    institution = edu.get('institution', '')
                    parts.append(f"  - {degree} from {institution}")
                else:
                    parts.append(f"  - {edu}")
        
        if data.get('certifications'):
            parts.append(f"Certifications: {', '.join(data['certifications'][:5])}")
        
        return "\n".join(parts)
    
    def _format_job_info(self, job) -> str:
        """Format job information for the prompt."""
        if hasattr(job, 'to_dict'):
            data = job.to_dict()
        elif isinstance(job, dict):
            data = job
        else:
            return str(job)
        
        parts = []
        
        if data.get('title'):
            parts.append(f"Title: {data['title']}")
        
        if data.get('company'):
            parts.append(f"Company: {data['company']}")
        
        if data.get('description'):
            parts.append(f"Description: {data['description'][:500]}")
        
        if data.get('required_skills'):
            skills = data['required_skills']
            if isinstance(skills, list):
                parts.append(f"Required Skills: {', '.join(skills)}")
            else:
                parts.append(f"Required Skills: {skills}")
        
        if data.get('preferred_skills'):
            skills = data['preferred_skills']
            if isinstance(skills, list):
                parts.append(f"Preferred Skills: {', '.join(skills)}")
        
        if data.get('required_experience'):
            parts.append(f"Experience Required: {data['required_experience']}")
        
        if data.get('education_requirements'):
            edu = data['education_requirements']
            if isinstance(edu, list):
                parts.append(f"Education: {', '.join(edu)}")
            else:
                parts.append(f"Education: {edu}")
        
        if data.get('responsibilities'):
            parts.append("Key Responsibilities:")
            for resp in data['responsibilities'][:5]:
                parts.append(f"  - {resp}")
        
        return "\n".join(parts)
    
    def _parse_response(self, response: str) -> MatchExplanation:
        """Parse LLM response into structured explanation."""
        # Default values
        summary = ""
        strengths = []
        gaps = []
        recommendations = []
        fit_reasoning = ""
        
        # Parse sections
        current_section = None
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            line_upper = line.upper()
            
            if line_upper.startswith('SUMMARY:'):
                summary = line[8:].strip()
                current_section = 'summary'
            elif line_upper.startswith('STRENGTHS:'):
                current_section = 'strengths'
            elif line_upper.startswith('GAPS:'):
                current_section = 'gaps'
            elif line_upper.startswith('RECOMMENDATIONS:'):
                current_section = 'recommendations'
            elif line_upper.startswith('FIT_SCORE_REASONING:') or line_upper.startswith('FIT SCORE REASONING:'):
                fit_reasoning = line.split(':', 1)[1].strip() if ':' in line else ""
                current_section = 'fit_reasoning'
            elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                # Bullet point
                item = line.lstrip('-•* ').strip()
                if item:
                    if current_section == 'strengths':
                        strengths.append(item)
                    elif current_section == 'gaps':
                        gaps.append(item)
                    elif current_section == 'recommendations':
                        recommendations.append(item)
            else:
                # Continuation of current section
                if current_section == 'summary' and not summary:
                    summary = line
                elif current_section == 'fit_reasoning':
                    fit_reasoning += " " + line
        
        # If parsing failed, use the whole response as summary
        if not summary and not strengths and not gaps:
            summary = response[:500] if len(response) > 500 else response
        
        return MatchExplanation(
            summary=summary.strip(),
            strengths=strengths[:5],
            gaps=gaps[:5],
            recommendations=recommendations[:3],
            fit_score_reasoning=fit_reasoning.strip(),
        )
    
    def explain_match(
        self,
        resume,
        job,
        similarity_score: float,
        skill_overlap: Dict[str, Any] = None
    ) -> MatchExplanation:
        """
        Generate an explanation for a resume-job match.
        
        Args:
            resume: ParsedResume object or dict
            job: ParsedJob object or dict
            similarity_score: Similarity score (0-1)
            skill_overlap: Optional skill overlap analysis
            
        Returns:
            MatchExplanation object
        """
        # Format inputs
        resume_info = self._format_resume_info(resume)
        job_info = self._format_job_info(job)
        
        # Add skill overlap if available
        if skill_overlap:
            job_info += f"\n\nSkill Analysis:"
            job_info += f"\n  - Matching Skills: {', '.join(skill_overlap.get('matching_skills', []))}"
            job_info += f"\n  - Missing Skills: {', '.join(skill_overlap.get('missing_skills', []))}"
            job_info += f"\n  - Coverage: {skill_overlap.get('coverage_percentage', 0):.1f}%"
        
        # Format prompt
        prompt = self.EXPLANATION_PROMPT.format(
            resume_info=resume_info,
            job_info=job_info,
            similarity_score=f"{similarity_score * 100:.1f}",
        )
        
        # Generate response
        try:
            if hasattr(self._llm, 'invoke'):
                response = self._llm.invoke(prompt)
                # Handle different response types
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
            else:
                response_text = self._llm(prompt)
        except Exception as e:
            # Fallback explanation if LLM fails
            return MatchExplanation(
                summary=f"Match score: {similarity_score*100:.1f}%. Unable to generate detailed explanation: {str(e)}",
                strengths=["Semantic similarity indicates potential match"],
                gaps=["Detailed analysis unavailable"],
                recommendations=["Review the resume and job description manually"],
                fit_score_reasoning="Score based on embedding similarity",
            )
        
        # Parse and return
        return self._parse_response(response_text)
    
    def explain_matches_batch(
        self,
        matches: List[Dict[str, Any]],
        resumes: List,
        jobs: List
    ) -> List[MatchExplanation]:
        """
        Generate explanations for multiple matches.
        
        Args:
            matches: List of match dictionaries with resume_index, job_index, similarity_score
            resumes: List of ParsedResume objects
            jobs: List of ParsedJob objects
            
        Returns:
            List of MatchExplanation objects
        """
        explanations = []
        
        for match in matches:
            resume_idx = match.get('resume_index', 0)
            job_idx = match.get('job_index', 0)
            score = match.get('similarity_score', 0)
            
            resume = resumes[resume_idx] if resume_idx < len(resumes) else {}
            job = jobs[job_idx] if job_idx < len(jobs) else {}
            
            explanation = self.explain_match(resume, job, score)
            explanations.append(explanation)
        
        return explanations
    
    def generate_quick_summary(self, resume, job, similarity_score: float) -> str:
        """
        Generate a quick one-sentence summary of a match.
        
        Args:
            resume: ParsedResume object or dict
            job: ParsedJob object or dict
            similarity_score: Similarity score (0-1)
            
        Returns:
            Quick summary string
        """
        # Get key info
        resume_data = resume.to_dict() if hasattr(resume, 'to_dict') else resume
        job_data = job.to_dict() if hasattr(job, 'to_dict') else job
        
        candidate_name = resume_data.get('name', 'This candidate')
        job_title = job_data.get('title', 'this position')
        
        resume_skills = set(s.lower() for s in resume_data.get('skills', []))
        job_skills = set(s.lower() for s in job_data.get('required_skills', []))
        
        matching = resume_skills & job_skills
        
        if similarity_score >= 0.8:
            fit = "excellent"
        elif similarity_score >= 0.6:
            fit = "good"
        elif similarity_score >= 0.4:
            fit = "moderate"
        else:
            fit = "limited"
        
        if matching:
            skills_text = f"with matching skills in {', '.join(list(matching)[:3])}"
        else:
            skills_text = "based on overall profile alignment"
        
        return f"{candidate_name} shows {fit} fit ({similarity_score*100:.0f}%) for {job_title} {skills_text}."
