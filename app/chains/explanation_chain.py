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
    experience_analysis: str = ""
    project_relevance: str = ""
    certification_education_fit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "strengths": self.strengths,
            "gaps": self.gaps,
            "recommendations": self.recommendations,
            "fit_score_reasoning": self.fit_score_reasoning,
            "experience_analysis": self.experience_analysis,
            "project_relevance": self.project_relevance,
            "certification_education_fit": self.certification_education_fit,
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
        
        if self.experience_analysis:
            parts.append(f"\n**Experience Analysis:** {self.experience_analysis}")
        
        if self.project_relevance:
            parts.append(f"\n**Project Relevance:** {self.project_relevance}")
        
        if self.certification_education_fit:
            parts.append(f"\n**Certification & Education Fit:** {self.certification_education_fit}")
        
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
    
    EXPLANATION_PROMPT = """You are an expert HR analyst performing a critical evaluation of a candidate-job match. Your analysis must be HONEST and THOROUGH, identifying both strengths AND disqualifying factors.

## CANDIDATE RESUME:
{resume_info}

## JOB REQUIREMENTS:
{job_info}

## MATCH ANALYSIS DATA:
{match_analysis}

## COMPUTED SIMILARITY SCORE: {similarity_score}%

---

## CRITICAL EVALUATION INSTRUCTIONS:

1. **HARD REQUIREMENTS CHECK**: First, identify if the job has any HARD requirements (PhD required, specific certifications required, minimum years of experience). If the candidate does NOT meet a hard requirement, this is a DISQUALIFYING factor that MUST be prominently mentioned.

2. **EDUCATION ANALYSIS**: Compare candidate's education level (Bachelor's/Master's/PhD) and field against the job's education requirements. A Master's degree does NOT satisfy a PhD requirement.

3. **SKILLS GAP ANALYSIS**: List specific required skills the candidate is MISSING. Be explicit about which critical skills are absent.

4. **EXPERIENCE EVALUATION**: Compare years of experience and relevance of past roles to the job requirements.

5. **CERTIFICATION CHECK**: Identify required certifications the candidate has vs. lacks.

---

Provide your analysis in this EXACT format:

SUMMARY: [2-3 sentences honestly assessing overall fit. Start with the most critical finding - if there's a disqualifying factor, mention it first.]

STRENGTHS:
- [specific strength with evidence from resume]
- [specific strength with evidence from resume]
- [specific strength with evidence from resume]

GAPS:
- [CRITICAL if disqualifying] [specific gap - e.g., "Education Mismatch: Candidate has MS, job requires PhD"]
- [specific missing skill or requirement]
- [specific missing skill or requirement]

EXPERIENCE_ANALYSIS: [1-2 sentences on experience fit - years, relevance, seniority]

PROJECT_RELEVANCE: [1 sentence on how candidate's projects align with job's tech stack/domain]

CERTIFICATION_EDUCATION_FIT: [1-2 sentences specifically comparing candidate's certs/education to job requirements. Be explicit about mismatches.]

RECOMMENDATION: [Strong Yes / Yes / Maybe / No / Strong No] - [1 sentence justification. If there are disqualifying factors like missing PhD, say "No" or "Strong No" regardless of other strengths.]

Be SPECIFIC, reference ACTUAL details from the resume, and be HONEST about gaps. Do not oversell the candidate."""

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
        
        model_name = self._kwargs.get("model") or os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
        
        self._llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )
        
        print(f"Initialized Google Generative AI: {model_name}")
    
    def _format_resume_info(self, resume) -> str:
        """Format resume information for the prompt with all fields."""
        if hasattr(resume, 'to_dict'):
            data = resume.to_dict()
        elif isinstance(resume, dict):
            data = resume
        else:
            return str(resume)
        
        parts = []
        
        if data.get('name'):
            parts.append(f"Name: {data['name']}")
        
        if data.get('location'):
            parts.append(f"Location: {data['location']}")
        
        if data.get('summary'):
            parts.append(f"Summary: {data['summary']}")
        
        if data.get('total_years_experience'):
            parts.append(f"Total Years of Experience: {data['total_years_experience']}")
        
        if data.get('skills'):
            skills = data['skills']
            if isinstance(skills, list):
                parts.append(f"Skills: {', '.join(skills[:20])}")
            else:
                parts.append(f"Skills: {skills}")
        
        if data.get('experience'):
            parts.append("\nWork Experience:")
            for i, exp in enumerate(data['experience'][:6], 1):
                if isinstance(exp, dict):
                    title = exp.get('title', 'Unknown')
                    company = exp.get('company', 'Unknown')
                    duration = exp.get('duration', '')
                    desc = exp.get('description', '')[:200]
                    achievements = exp.get('achievements', [])
                    
                    exp_str = f"  {i}. {title} at {company}"
                    if duration:
                        exp_str += f" ({duration})"
                    parts.append(exp_str)
                    if desc:
                        parts.append(f"     Description: {desc}")
                    if achievements:
                        parts.append(f"     Key Achievements: {'; '.join(achievements[:3])}")
                else:
                    parts.append(f"  {i}. {exp}")
        
        if data.get('projects'):
            parts.append("\nProjects:")
            for i, proj in enumerate(data['projects'][:5], 1):
                if isinstance(proj, dict):
                    name = proj.get('name', 'Unknown')
                    desc = proj.get('description', '')[:150]
                    techs = proj.get('technologies', [])
                    highlights = proj.get('highlights', [])
                    
                    parts.append(f"  {i}. {name}")
                    if desc:
                        parts.append(f"     Description: {desc}")
                    if techs:
                        parts.append(f"     Technologies: {', '.join(techs)}")
                    if highlights:
                        parts.append(f"     Highlights: {', '.join(highlights)}")
                else:
                    parts.append(f"  {i}. {proj}")
        
        if data.get('education'):
            parts.append("\nEducation:")
            for edu in data['education'][:3]:
                if isinstance(edu, dict):
                    degree = edu.get('degree', '')
                    field = edu.get('field', '')
                    institution = edu.get('institution', '')
                    year = edu.get('year', '')
                    gpa = edu.get('gpa', '')
                    honors = edu.get('honors', '')
                    
                    edu_str = f"  - {degree}"
                    if field:
                        edu_str += f" in {field}"
                    if institution:
                        edu_str += f" from {institution}"
                    if year:
                        edu_str += f" ({year})"
                    parts.append(edu_str)
                    if gpa:
                        parts.append(f"    GPA: {gpa}")
                    if honors:
                        parts.append(f"    Honors: {honors}")
                else:
                    parts.append(f"  - {edu}")
        
        if data.get('certifications'):
            parts.append("\nCertifications:")
            for cert in data['certifications'][:6]:
                if isinstance(cert, dict):
                    name = cert.get('name', '')
                    issuer = cert.get('issuer', '')
                    year = cert.get('year', '')
                    cert_str = f"  - {name}"
                    if issuer:
                        cert_str += f" (by {issuer})"
                    if year:
                        cert_str += f" - {year}"
                    parts.append(cert_str)
                else:
                    parts.append(f"  - {cert}")
        
        if data.get('publications'):
            parts.append("\nPublications:")
            for pub in data['publications'][:3]:
                if isinstance(pub, dict):
                    title = pub.get('title', '')
                    venue = pub.get('venue', '')
                    year = pub.get('year', '')
                    pub_str = f"  - {title}"
                    if venue:
                        pub_str += f" ({venue})"
                    if year:
                        pub_str += f" - {year}"
                    parts.append(pub_str)
                else:
                    parts.append(f"  - {pub}")
        
        if data.get('awards'):
            parts.append(f"\nAwards: {', '.join(data['awards'][:5])}")
        
        if data.get('interests'):
            parts.append(f"\nInterests: {', '.join(data['interests'][:8])}")
        
        if data.get('languages'):
            parts.append(f"\nLanguages: {', '.join(data['languages'])}")
        
        return "\n".join(parts)
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
        """Format job information for the prompt with all fields."""
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
        
        if data.get('location'):
            location = data['location']
            if data.get('remote_type'):
                location += f" ({data['remote_type']})"
            parts.append(f"Location: {location}")
        
        if data.get('job_type'):
            parts.append(f"Job Type: {data['job_type']}")
        
        if data.get('industry'):
            parts.append(f"Industry: {data['industry']}")
        
        if data.get('description'):
            parts.append(f"\nDescription: {data['description'][:600]}")
        
        if data.get('required_skills'):
            skills = data['required_skills']
            if isinstance(skills, list):
                parts.append(f"\nRequired Skills: {', '.join(skills)}")
            else:
                parts.append(f"\nRequired Skills: {skills}")
        
        if data.get('preferred_skills'):
            skills = data['preferred_skills']
            if isinstance(skills, list):
                parts.append(f"Preferred Skills: {', '.join(skills)}")
        
        if data.get('tech_stack'):
            parts.append(f"Tech Stack: {', '.join(data['tech_stack'][:15])}")
        
        if data.get('required_experience'):
            parts.append(f"\nExperience Required: {data['required_experience']}")
        elif data.get('min_years_experience'):
            exp_str = f"{data['min_years_experience']}+"
            if data.get('max_years_experience'):
                exp_str = f"{data['min_years_experience']}-{data['max_years_experience']}"
            parts.append(f"\nExperience Required: {exp_str} years")
        
        if data.get('min_education_level'):
            parts.append(f"Minimum Education: {data['min_education_level']}")
        
        if data.get('education_requirements'):
            edu = data['education_requirements']
            if isinstance(edu, list):
                parts.append(f"Education Requirements: {', '.join(edu)}")
            else:
                parts.append(f"Education Requirements: {edu}")
        
        if data.get('required_certifications'):
            parts.append(f"\nRequired Certifications: {', '.join(data['required_certifications'])}")
        
        if data.get('preferred_certifications'):
            parts.append(f"Preferred Certifications: {', '.join(data['preferred_certifications'])}")
        
        if data.get('project_types'):
            parts.append(f"\nProject Types: {', '.join(data['project_types'])}")
        
        if data.get('responsibilities'):
            parts.append("\nKey Responsibilities:")
            for resp in data['responsibilities'][:6]:
                parts.append(f"  - {resp}")
        
        if data.get('salary_range'):
            parts.append(f"\nSalary Range: {data['salary_range']}")
        
        if data.get('benefits'):
            parts.append(f"Benefits: {', '.join(data['benefits'][:5])}")
        
        return "\n".join(parts)
    
    def _parse_response(self, response: str) -> MatchExplanation:
        """Parse LLM response into structured explanation."""
        # Default values
        summary = ""
        strengths = []
        gaps = []
        recommendations = []
        fit_reasoning = ""
        experience_analysis = ""
        project_relevance = ""
        certification_education_fit = ""
        
        # Handle empty or very short responses
        if not response or len(response.strip()) < 20:
            return MatchExplanation(
                summary="Unable to generate detailed explanation.",
                strengths=["Semantic similarity indicates potential match"],
                gaps=["Detailed analysis unavailable"],
                recommendations=["Review the resume and job description manually"],
                fit_score_reasoning="Score based on automated analysis",
            )
        
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
            elif 'STRENGTHS' in line_upper and ':' in line:
                current_section = 'strengths'
            elif 'GAPS' in line_upper and ':' in line:
                current_section = 'gaps'
            elif 'RECOMMENDATION' in line_upper and ':' in line:
                # Handle both RECOMMENDATIONS: and RECOMMENDATION: [text]
                after_colon = line.split(':', 1)[1].strip() if ':' in line else ""
                if after_colon and not after_colon.startswith('-'):
                    fit_reasoning = after_colon
                current_section = 'recommendations'
            elif line_upper.startswith('EXPERIENCE_ANALYSIS:') or line_upper.startswith('EXPERIENCE ANALYSIS:'):
                experience_analysis = line.split(':', 1)[1].strip() if ':' in line else ""
                current_section = 'experience_analysis'
            elif line_upper.startswith('PROJECT_RELEVANCE:') or line_upper.startswith('PROJECT RELEVANCE:'):
                project_relevance = line.split(':', 1)[1].strip() if ':' in line else ""
                current_section = 'project_relevance'
            elif 'CERTIFICATION' in line_upper or 'EDUCATION FIT' in line_upper:
                certification_education_fit = line.split(':', 1)[1].strip() if ':' in line else ""
                current_section = 'certification_education_fit'
            elif line_upper.startswith('FIT_SCORE_REASONING:') or line_upper.startswith('FIT SCORE REASONING:'):
                fit_reasoning = line.split(':', 1)[1].strip() if ':' in line else ""
                current_section = 'fit_reasoning'
            elif line.startswith('-') or line.startswith('•') or line.startswith('*') or (len(line) > 2 and line[0].isdigit() and line[1] in '.):'):
                # Bullet point or numbered item
                item = line.lstrip('-•*0123456789.): ').strip()
                if item:
                    if current_section == 'strengths':
                        strengths.append(item)
                    elif current_section == 'gaps':
                        gaps.append(item)
                    elif current_section == 'recommendations':
                        recommendations.append(item)
            else:
                # Continuation of current section
                if current_section == 'summary' and line and not summary:
                    summary = line
                elif current_section == 'summary' and line:
                    summary += " " + line
                elif current_section == 'fit_reasoning':
                    fit_reasoning += " " + line
                elif current_section == 'experience_analysis':
                    experience_analysis += " " + line
                elif current_section == 'project_relevance':
                    project_relevance += " " + line
                elif current_section == 'certification_education_fit':
                    certification_education_fit += " " + line
        
        # If parsing failed, try to extract something useful
        if not summary and not strengths and not gaps:
            # Use first paragraph as summary
            paragraphs = response.split('\n\n')
            if paragraphs:
                summary = paragraphs[0].strip()[:500]
            else:
                summary = response[:500]
        
        return MatchExplanation(
            summary=summary.strip(),
            strengths=strengths[:6] if strengths else ["Analysis generated - see summary"],
            gaps=gaps[:5] if gaps else [],
            recommendations=recommendations[:4] if recommendations else [],
            fit_score_reasoning=fit_reasoning.strip(),
            experience_analysis=experience_analysis.strip(),
            project_relevance=project_relevance.strip(),
            certification_education_fit=certification_education_fit.strip(),
        )
    
    def _format_match_analysis(self, match_analysis: Dict[str, Any]) -> str:
        """Format comprehensive match analysis for the prompt."""
        if not match_analysis:
            return "No detailed analysis available."
        
        parts = []
        
        # Component scores
        if match_analysis.get('component_scores'):
            scores = match_analysis['component_scores']
            parts.append("Component Scores:")
            for component, score in scores.items():
                parts.append(f"  - {component.title()}: {score*100:.1f}%")
        
        # Skill analysis - CRITICAL for gaps
        if match_analysis.get('skill_analysis'):
            skill = match_analysis['skill_analysis']
            parts.append(f"\n=== SKILL ANALYSIS ===")
            parts.append(f"Skill Coverage: {skill.get('coverage_percentage', 0):.1f}%")
            if skill.get('matching_skills'):
                parts.append(f"  MATCHING Skills: {', '.join(skill['matching_skills'][:15])}")
            if skill.get('missing_skills'):
                parts.append(f"  MISSING Skills (GAPS): {', '.join(skill['missing_skills'][:15])}")
        
        # Experience analysis
        if match_analysis.get('experience_analysis'):
            exp = match_analysis['experience_analysis']
            parts.append(f"\n=== EXPERIENCE ANALYSIS ===")
            parts.append(f"Candidate Years: {exp.get('candidate_years', exp.get('total_years', 'N/A'))}")
            parts.append(f"Required Years: {exp.get('required_min', exp.get('required_years', 'N/A'))} - {exp.get('required_max', 'N/A')}")
            if exp.get('analysis'):
                parts.append(f"Analysis: {exp.get('analysis')}")
        
        # Also check directly passed candidate experience
        if match_analysis.get('candidate_experience_years'):
            parts.append(f"Candidate Total Experience: {match_analysis['candidate_experience_years']} years")
        if match_analysis.get('required_experience'):
            parts.append(f"Job Required Experience: {match_analysis['required_experience']}")
        
        # Education analysis - CRITICAL FOR PhD/MS MISMATCHES
        parts.append(f"\n=== EDUCATION ANALYSIS (CHECK FOR HARD REQUIREMENTS) ===")
        if match_analysis.get('education_analysis'):
            edu = match_analysis['education_analysis']
            parts.append(f"Candidate Education: {edu.get('candidate_level', 'N/A')} in {edu.get('candidate_field', 'N/A')}")
            parts.append(f"Required Education: {edu.get('required_level', 'N/A')}")
            if edu.get('analysis'):
                parts.append(f"Education Analysis: {edu.get('analysis')}")
            # Explicit mismatch flag
            cand_level = str(edu.get('candidate_level', '')).lower()
            req_level = str(edu.get('required_level', '')).lower()
            if 'phd' in req_level and 'phd' not in cand_level:
                parts.append("*** CRITICAL: JOB REQUIRES PhD BUT CANDIDATE DOES NOT HAVE ONE ***")
        
        # Also check directly passed education
        if match_analysis.get('candidate_education'):
            cand_edu = match_analysis['candidate_education']
            if isinstance(cand_edu, list) and cand_edu:
                for edu in cand_edu[:2]:
                    if isinstance(edu, dict):
                        parts.append(f"Candidate Degree: {edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')}")
        if match_analysis.get('required_education'):
            parts.append(f"Job Minimum Education: {match_analysis['required_education']}")
            # Check for PhD requirement mismatch
            req_edu = str(match_analysis.get('required_education', '')).lower()
            if 'phd' in req_edu or 'doctorate' in req_edu:
                parts.append("*** NOTE: This job has a PhD/Doctorate requirement - verify candidate meets this ***")
        
        # Certification analysis
        if match_analysis.get('certification_analysis'):
            cert = match_analysis['certification_analysis']
            parts.append(f"\n=== CERTIFICATION ANALYSIS ===")
            parts.append(f"Required Certification Coverage: {cert.get('required_coverage_percentage', 0):.1f}%")
            if cert.get('matched_certs'):
                parts.append(f"  MATCHED Certifications: {', '.join(cert['matched_certs'])}")
            if cert.get('missing_required'):
                parts.append(f"  MISSING Required Certifications: {', '.join(cert['missing_required'])}")
        
        # Also check directly passed certifications
        if match_analysis.get('candidate_certifications'):
            certs = match_analysis['candidate_certifications']
            cert_names = [c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in certs[:8]]
            parts.append(f"Candidate Certifications: {', '.join(cert_names)}")
        if match_analysis.get('required_certifications'):
            parts.append(f"Job Required Certifications: {', '.join(match_analysis['required_certifications'])}")
        
        # Project analysis
        if match_analysis.get('project_analysis'):
            proj = match_analysis['project_analysis']
            parts.append(f"\n=== PROJECT RELEVANCE ===")
            if proj.get('analysis'):
                parts.append(f"Analysis: {proj.get('analysis')}")
            if proj.get('relevant_projects'):
                parts.append("Relevant Projects:")
                for rp in proj['relevant_projects'][:3]:
                    parts.append(f"  - {rp.get('name', 'Unknown')}: {', '.join(rp.get('matching_techs', []))}")
        
        return "\n".join(parts)
        
        return "\n".join(parts)
    
    def explain_match(
        self,
        resume,
        job,
        similarity_score: float,
        skill_overlap: Dict[str, Any] = None,
        match_analysis: Dict[str, Any] = None
    ) -> MatchExplanation:
        """
        Generate an explanation for a resume-job match.
        
        Args:
            resume: ParsedResume object or dict
            job: ParsedJob object or dict
            similarity_score: Similarity score (0-1)
            skill_overlap: Optional skill overlap analysis (deprecated, use match_analysis)
            match_analysis: Comprehensive match analysis from compute_comprehensive_match
            
        Returns:
            MatchExplanation object
        """
        # Format inputs
        resume_info = self._format_resume_info(resume)
        job_info = self._format_job_info(job)
        
        # Format match analysis
        analysis_text = self._format_match_analysis(match_analysis) if match_analysis else ""
        
        # Add skill overlap if provided (backward compatibility)
        if skill_overlap and not match_analysis:
            analysis_text = "Skill Analysis:"
            analysis_text += f"\n  - Matching Skills: {', '.join(skill_overlap.get('matching_skills', []))}"
            analysis_text += f"\n  - Missing Skills: {', '.join(skill_overlap.get('missing_skills', []))}"
            analysis_text += f"\n  - Coverage: {skill_overlap.get('coverage_percentage', 0):.1f}%"
        
        # Format prompt
        prompt = self.EXPLANATION_PROMPT.format(
            resume_info=resume_info,
            job_info=job_info,
            match_analysis=analysis_text or "No detailed analysis available.",
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
                experience_analysis="",
                project_relevance="",
                certification_education_fit="",
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
