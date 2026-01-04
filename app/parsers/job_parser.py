"""
Job Description Parser Module

Parses job postings from files or text to extract structured information
including title, company, requirements, skills, and qualifications.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedJob:
    """Structured representation of a parsed job description."""
    raw_text: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    job_type: str = ""  # Full-time, Part-time, Contract, etc.
    remote_type: str = ""  # Remote, Hybrid, On-site
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    required_experience: str = ""
    min_years_experience: int = 0
    max_years_experience: int = 0
    education_requirements: List[str] = field(default_factory=list)
    min_education_level: str = ""  # Bachelor's, Master's, PhD
    required_certifications: List[str] = field(default_factory=list)
    preferred_certifications: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    salary_range: str = ""
    salary_min: int = 0
    salary_max: int = 0
    description: str = ""
    industry: str = ""
    team_size: str = ""
    tech_stack: List[str] = field(default_factory=list)
    project_types: List[str] = field(default_factory=list)
    file_name: str = ""
    
    def to_text_for_embedding(self) -> str:
        """
        Generate a text representation optimized for embedding generation.
        
        Returns:
            Concatenated text of all job sections
        """
        parts = []
        
        if self.title:
            parts.append(f"Job Title: {self.title}")
        
        if self.company:
            parts.append(f"Company: {self.company}")
        
        if self.description:
            parts.append(f"Description: {self.description}")
        
        if self.required_skills:
            parts.append(f"Required Skills: {', '.join(self.required_skills)}")
        
        if self.preferred_skills:
            parts.append(f"Preferred Skills: {', '.join(self.preferred_skills)}")
        
        if self.tech_stack:
            parts.append(f"Tech Stack: {', '.join(self.tech_stack)}")
        
        if self.responsibilities:
            parts.append(f"Responsibilities: {'; '.join(self.responsibilities)}")
        
        if self.education_requirements:
            parts.append(f"Education: {', '.join(self.education_requirements)}")
        
        if self.required_experience:
            parts.append(f"Experience Required: {self.required_experience}")
        
        if self.required_certifications:
            parts.append(f"Required Certifications: {', '.join(self.required_certifications)}")
        
        if self.preferred_certifications:
            parts.append(f"Preferred Certifications: {', '.join(self.preferred_certifications)}")
        
        if self.project_types:
            parts.append(f"Project Types: {', '.join(self.project_types)}")
        
        if self.industry:
            parts.append(f"Industry: {self.industry}")
        
        # Include raw text if structured extraction is limited
        if len(parts) < 3 and self.raw_text:
            parts.append(self.raw_text[:2000])
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "job_type": self.job_type,
            "remote_type": self.remote_type,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "required_experience": self.required_experience,
            "min_years_experience": self.min_years_experience,
            "max_years_experience": self.max_years_experience,
            "education_requirements": self.education_requirements,
            "min_education_level": self.min_education_level,
            "required_certifications": self.required_certifications,
            "preferred_certifications": self.preferred_certifications,
            "responsibilities": self.responsibilities,
            "benefits": self.benefits,
            "salary_range": self.salary_range,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "description": self.description,
            "industry": self.industry,
            "team_size": self.team_size,
            "tech_stack": self.tech_stack,
            "project_types": self.project_types,
            "file_name": self.file_name,
        }


class JobParser:
    """
    Parser for extracting structured information from job descriptions.
    
    Supports text input, PDF, DOCX, and TXT files.
    """
    
    # Common skills for job matching
    TECH_SKILLS = [
        # Programming
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
        "rust", "php", "swift", "kotlin", "scala", "r", "sql", "html", "css",
        # Frameworks
        "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
        "spring", "express", ".net", "rails",
        # Data/ML
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "machine learning", "deep learning", "nlp", "computer vision",
        "data science", "data analysis", "data engineering", "statistics",
        "langchain", "langgraph", "llm", "generative ai",
        # Cloud/DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
        "ci/cd", "git", "linux",
        # Databases
        "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle",
        # Tools
        "jira", "confluence", "slack", "tableau", "power bi", "excel",
        # Methodologies
        "agile", "scrum", "kanban", "devops", "microservices", "rest api",
    ]
    
    # Section header patterns
    SECTION_PATTERNS = {
        "responsibilities": r"(?i)(responsibilities|duties|what you.?ll do|key responsibilities|role|your role)",
        "requirements": r"(?i)(requirements|qualifications|what we.?re looking for|must have|required)",
        "preferred": r"(?i)(preferred|nice to have|bonus|ideal|desired)",
        "benefits": r"(?i)(benefits|perks|what we offer|compensation|why join)",
        "about": r"(?i)(about us|about the company|who we are|company overview)",
        "skills": r"(?i)(skills|technical skills|competencies|technologies|tech stack)",
        "certifications": r"(?i)(certifications?|certificates?|credentials)",
        "projects": r"(?i)(projects?|project\s*types?|what you.?ll work on)",
    }
    
    def __init__(self):
        """Initialize the job parser."""
        # Import resume parser for file handling
        from .resume_parser import ResumeParser
        self._file_parser = ResumeParser()
    
    def parse(self, text: str = None, file_path: str = None, 
              file_content: bytes = None, file_name: str = None) -> ParsedJob:
        """
        Parse a job description and extract structured information.
        
        Args:
            text: Raw job description text
            file_path: Path to the job description file
            file_content: Raw file content (bytes)
            file_name: Original filename
            
        Returns:
            ParsedJob object with extracted information
        """
        # Get raw text from input
        if text:
            raw_text = text
            file_name = file_name or "direct_input"
        elif file_path or file_content:
            # Use resume parser's file reading capabilities
            if file_path:
                ext = Path(file_path).suffix.lower()
                file_name = file_name or Path(file_path).name
            else:
                ext = Path(file_name).suffix.lower() if file_name else ".txt"
            
            if ext == ".pdf":
                raw_text = self._file_parser._parse_pdf(file_path, file_content)
            elif ext == ".docx":
                raw_text = self._file_parser._parse_docx(file_path, file_content)
            else:
                raw_text = self._file_parser._parse_txt(file_path, file_content)
        else:
            raise ValueError("Either text, file_path, or file_content must be provided")
        
        # Create parsed job object
        job = ParsedJob(raw_text=raw_text, file_name=file_name or "unknown")
        
        # Extract structured information
        job.title = self._extract_title(raw_text)
        job.company = self._extract_company(raw_text)
        job.location = self._extract_location(raw_text)
        job.job_type = self._extract_job_type(raw_text)
        job.remote_type = self._extract_remote_type(raw_text)
        job.required_skills = self._extract_required_skills(raw_text)
        job.preferred_skills = self._extract_preferred_skills(raw_text)
        job.required_experience = self._extract_experience_requirement(raw_text)
        job.min_years_experience, job.max_years_experience = self._extract_experience_years(raw_text)
        job.education_requirements = self._extract_education_requirements(raw_text)
        job.min_education_level = self._extract_min_education_level(raw_text)
        job.required_certifications = self._extract_required_certifications(raw_text)
        job.preferred_certifications = self._extract_preferred_certifications(raw_text)
        job.responsibilities = self._extract_responsibilities(raw_text)
        job.benefits = self._extract_benefits(raw_text)
        job.salary_range = self._extract_salary(raw_text)
        job.salary_min, job.salary_max = self._extract_salary_range(raw_text)
        job.description = self._extract_description(raw_text)
        job.industry = self._extract_industry(raw_text)
        job.team_size = self._extract_team_size(raw_text)
        job.tech_stack = self._extract_tech_stack(raw_text)
        job.project_types = self._extract_project_types(raw_text)
        
        return job
    
    def parse_multiple(self, texts: List[str] = None, 
                       file_paths: List[str] = None) -> List[ParsedJob]:
        """
        Parse multiple job descriptions.
        
        Args:
            texts: List of raw job description texts
            file_paths: List of paths to job description files
            
        Returns:
            List of ParsedJob objects
        """
        jobs = []
        
        if texts:
            for text in texts:
                jobs.append(self.parse(text=text))
        
        if file_paths:
            for path in file_paths:
                jobs.append(self.parse(file_path=path))
        
        return jobs
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract content of a specific section."""
        pattern = self.SECTION_PATTERNS.get(section_name)
        if not pattern:
            return ""
        
        # Find section header
        match = re.search(pattern, text)
        if not match:
            return ""
        
        start = match.end()
        
        # Find next section header or end of text
        remaining_text = text[start:]
        min_pos = len(remaining_text)
        
        for other_pattern in self.SECTION_PATTERNS.values():
            if other_pattern != pattern:
                other_match = re.search(other_pattern, remaining_text)
                if other_match and other_match.start() < min_pos and other_match.start() > 20:
                    min_pos = other_match.start()
        
        return remaining_text[:min_pos].strip()
    
    def _extract_title(self, text: str) -> str:
        """Extract job title."""
        lines = text.strip().split('\n')
        
        # Common job title patterns
        title_patterns = [
            r"(?i)(?:job\s*title|position|role)\s*[:\-]?\s*(.+)",
            r"(?i)^((?:Senior|Junior|Lead|Principal|Staff)?\s*[A-Za-z\s]+(?:Engineer|Developer|Manager|Analyst|Designer|Scientist|Architect|Consultant|Specialist))",
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text[:500])
            if match:
                title = match.group(1).strip()
                if len(title) < 100:
                    return title
        
        # Fallback: first non-empty line that looks like a title
        for line in lines[:5]:
            line = line.strip()
            if line and 5 < len(line) < 80:
                if not any(kw in line.lower() for kw in ['http', '@', 'apply', 'posted']):
                    return line
        
        return ""
    
    def _extract_company(self, text: str) -> str:
        """Extract company name."""
        patterns = [
            r"(?i)(?:company|employer|organization)\s*[:\-]?\s*(.+?)(?:\n|$)",
            r"(?i)(?:at|@)\s+([A-Z][A-Za-z0-9\s&.,]+?)(?:\s+(?:is|we|are)|$|\n)",
            r"(?i)([A-Z][A-Za-z0-9\s&]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:1000])
            if match:
                company = match.group(1).strip()
                if 2 < len(company) < 100:
                    return company
        
        return ""
    
    def _extract_location(self, text: str) -> str:
        """Extract job location."""
        patterns = [
            r"(?i)(?:location|based in|office)\s*[:\-]?\s*(.+?)(?:\n|$)",
            r"([A-Z][a-z]+(?:,\s*[A-Z]{2})?)\s*(?:\(remote\)|remote|hybrid)?",
            r"(?i)(remote|hybrid|on-?site|work from home)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:500])
            if match:
                location = match.group(1).strip()
                if 2 < len(location) < 100:
                    return location
        
        return ""
    
    def _extract_job_type(self, text: str) -> str:
        """Extract job type (full-time, part-time, etc.)."""
        patterns = [
            r"(?i)(full[- ]?time|part[- ]?time|contract|temporary|freelance|intern(?:ship)?)",
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).title()
        
        return "Full-time"  # Default assumption
    
    def _extract_required_skills(self, text: str) -> List[str]:
        """Extract required skills."""
        skills = []
        text_lower = text.lower()
        
        # Get requirements section
        req_section = self._extract_section(text, "requirements")
        skills_section = self._extract_section(text, "skills")
        search_text = (req_section + " " + skills_section + " " + text).lower()
        
        # Find skills from predefined list
        for skill in self.TECH_SKILLS:
            if skill in search_text:
                formatted = skill.title() if len(skill) > 3 else skill.upper()
                if formatted not in skills:
                    skills.append(formatted)
        
        return skills[:20]
    
    def _extract_preferred_skills(self, text: str) -> List[str]:
        """Extract preferred/nice-to-have skills."""
        preferred = []
        
        pref_section = self._extract_section(text, "preferred")
        if not pref_section:
            return preferred
        
        text_lower = pref_section.lower()
        
        for skill in self.TECH_SKILLS:
            if skill in text_lower:
                formatted = skill.title() if len(skill) > 3 else skill.upper()
                if formatted not in preferred:
                    preferred.append(formatted)
        
        return preferred[:10]
    
    def _extract_experience_requirement(self, text: str) -> str:
        """Extract years of experience requirement."""
        patterns = [
            r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
            r"(?:minimum|at least)\s*(\d+)\s*(?:years?|yrs?)",
            r"(\d+)\s*(?:-|to)\s*(\d+)\s*(?:years?|yrs?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:
                    return f"{groups[0]}-{groups[1]} years"
                return f"{groups[0]}+ years"
        
        return ""
    
    def _extract_education_requirements(self, text: str) -> List[str]:
        """Extract education requirements."""
        education = []
        
        patterns = [
            r"(?i)(Bachelor'?s?|B\.?S\.?|B\.?A\.?)\s*(?:degree)?\s*(?:in\s+)?([A-Za-z\s]+)?",
            r"(?i)(Master'?s?|M\.?S\.?|M\.?A\.?|MBA)\s*(?:degree)?\s*(?:in\s+)?([A-Za-z\s]+)?",
            r"(?i)(Ph\.?D\.?|Doctorate)\s*(?:in\s+)?([A-Za-z\s]+)?",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                degree = match.group(1)
                field = match.group(2).strip() if match.group(2) else ""
                edu_str = f"{degree}" + (f" in {field}" if field and len(field) < 50 else "")
                if edu_str not in education:
                    education.append(edu_str)
        
        return education[:5]
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract job responsibilities."""
        responsibilities = []
        
        resp_section = self._extract_section(text, "responsibilities")
        if not resp_section:
            return responsibilities
        
        # Split by bullet points or newlines
        items = re.split(r'[•\-\*\n]', resp_section)
        
        for item in items:
            item = item.strip()
            if item and 10 < len(item) < 300:
                responsibilities.append(item)
        
        return responsibilities[:15]
    
    def _extract_benefits(self, text: str) -> List[str]:
        """Extract job benefits."""
        benefits = []
        
        benefits_section = self._extract_section(text, "benefits")
        if not benefits_section:
            return benefits
        
        # Split by bullet points or newlines
        items = re.split(r'[•\-\*\n]', benefits_section)
        
        for item in items:
            item = item.strip()
            if item and 5 < len(item) < 200:
                benefits.append(item)
        
        return benefits[:10]
    
    def _extract_salary(self, text: str) -> str:
        """Extract salary range if mentioned."""
        patterns = [
            r"\$(\d{2,3}(?:,\d{3})?(?:k|K)?)\s*(?:-|to)\s*\$?(\d{2,3}(?:,\d{3})?(?:k|K)?)",
            r"(?i)salary\s*[:\-]?\s*\$?(\d{2,3}(?:,\d{3})?(?:k|K)?)",
            r"(\d{2,3}(?:,\d{3})?)\s*(?:-|to)\s*(\d{2,3}(?:,\d{3})?)\s*(?:per year|annually|/year)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:
                    return f"${groups[0]} - ${groups[1]}"
                return f"${groups[0]}"
        
        return ""
    
    def _extract_description(self, text: str) -> str:
        """Extract main job description."""
        about_section = self._extract_section(text, "about")
        
        if about_section:
            return about_section[:1000]
        
        # Return first significant paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) > 100:
                return para[:1000]
        
        return text[:1000] if text else ""

    def _extract_remote_type(self, text: str) -> str:
        """Extract remote work type (Remote, Hybrid, On-site)."""
        text_lower = text.lower()
        
        if "fully remote" in text_lower or "100% remote" in text_lower or "remote first" in text_lower:
            return "Remote"
        elif "hybrid" in text_lower:
            return "Hybrid"
        elif "on-site" in text_lower or "onsite" in text_lower or "in-office" in text_lower:
            return "On-site"
        elif "remote" in text_lower:
            return "Remote"
        
        return ""
    
    def _extract_experience_years(self, text: str) -> tuple:
        """Extract minimum and maximum years of experience."""
        patterns = [
            r"(\d+)\s*(?:-|to)\s*(\d+)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
            r"(\d+)\+\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
            r"(?:minimum|at least)\s*(\d+)\s*(?:years?|yrs?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:
                    return (int(groups[0]), int(groups[1]))
                elif len(groups) >= 1 and groups[0]:
                    years = int(groups[0])
                    return (years, years + 5)  # Assume range of 5 years above minimum
        
        return (0, 0)
    
    def _extract_min_education_level(self, text: str) -> str:
        """Extract minimum education level required."""
        text_lower = text.lower()
        
        if "ph.d" in text_lower or "phd" in text_lower or "doctorate" in text_lower:
            return "PhD"
        elif "master" in text_lower or "m.s." in text_lower or "m.a." in text_lower or "mba" in text_lower:
            return "Master's"
        elif "bachelor" in text_lower or "b.s." in text_lower or "b.a." in text_lower:
            return "Bachelor's"
        elif "associate" in text_lower:
            return "Associate's"
        elif "high school" in text_lower or "ged" in text_lower:
            return "High School"
        
        return ""
    
    def _extract_required_certifications(self, text: str) -> List[str]:
        """Extract required certifications from job description."""
        certifications = []
        
        # Get requirements section
        req_section = self._extract_section(text, "requirements")
        cert_section = self._extract_section(text, "certifications")
        search_text = (req_section + " " + cert_section).lower() if req_section or cert_section else text.lower()
        
        # Common certification patterns
        cert_patterns = [
            r"(AWS\s+(?:Certified\s+)?[A-Za-z\s]+(?:Architect|Developer|Administrator|Engineer|Practitioner))",
            r"(Azure\s+(?:Certified\s+)?[A-Za-z\s]+(?:Administrator|Developer|Architect|Engineer))",
            r"(Google\s+(?:Cloud\s+)?(?:Certified\s+)?[A-Za-z\s]+(?:Engineer|Architect|Developer))",
            r"(Certified\s+[A-Za-z\s]+(?:Professional|Associate|Expert))",
            r"(PMP|CISSP|CPA|CFA|CCNA|CCNP|CompTIA\s+[A-Za-z+]+|TOGAF|Six\s+Sigma|Scrum\s+Master|ITIL)",
            r"(TensorFlow\s+(?:Developer\s+)?Certificate)",
        ]
        
        for pattern in cert_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                cert = match.group(1).strip()
                if cert and cert not in certifications:
                    certifications.append(cert)
        
        return certifications[:10]
    
    def _extract_preferred_certifications(self, text: str) -> List[str]:
        """Extract preferred/nice-to-have certifications."""
        certifications = []
        
        pref_section = self._extract_section(text, "preferred")
        if not pref_section:
            return certifications
        
        cert_patterns = [
            r"(AWS\s+(?:Certified\s+)?[A-Za-z\s]+(?:Architect|Developer|Administrator|Engineer|Practitioner))",
            r"(Azure\s+(?:Certified\s+)?[A-Za-z\s]+)",
            r"(Google\s+(?:Cloud\s+)?(?:Certified\s+)?[A-Za-z\s]+)",
            r"(Certified\s+[A-Za-z\s]+(?:Professional|Associate|Expert))",
            r"(PMP|CISSP|CPA|CFA|CCNA|CCNP|CompTIA\s+[A-Za-z+]+)",
        ]
        
        for pattern in cert_patterns:
            matches = re.finditer(pattern, pref_section, re.IGNORECASE)
            for match in matches:
                cert = match.group(1).strip()
                if cert and cert not in certifications:
                    certifications.append(cert)
        
        return certifications[:10]
    
    def _extract_salary_range(self, text: str) -> tuple:
        """Extract salary min and max as integers."""
        patterns = [
            r"\$(\d{2,3}(?:,\d{3})?)\s*(?:k|K)?\s*(?:-|to)\s*\$?(\d{2,3}(?:,\d{3})?)\s*(?:k|K)?",
            r"(\d{2,3}(?:,\d{3})?)\s*(?:-|to)\s*(\d{2,3}(?:,\d{3})?)\s*(?:per year|annually|/year)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    min_sal = int(match.group(1).replace(",", ""))
                    max_sal = int(match.group(2).replace(",", ""))
                    
                    # Handle 'k' notation (e.g., 150k)
                    if min_sal < 1000:
                        min_sal *= 1000
                    if max_sal < 1000:
                        max_sal *= 1000
                    
                    return (min_sal, max_sal)
                except (ValueError, AttributeError):
                    pass
        
        return (0, 0)
    
    def _extract_industry(self, text: str) -> str:
        """Extract industry/domain from job description."""
        industries = {
            "healthcare": ["healthcare", "medical", "pharma", "hospital", "clinical", "biotech"],
            "finance": ["finance", "banking", "fintech", "investment", "trading", "insurance"],
            "technology": ["software", "saas", "tech company", "startup", "technology"],
            "e-commerce": ["e-commerce", "retail", "marketplace", "shopping"],
            "education": ["education", "edtech", "learning", "university", "school"],
            "manufacturing": ["manufacturing", "industrial", "factory", "production"],
            "media": ["media", "entertainment", "streaming", "content", "publishing"],
            "consulting": ["consulting", "advisory", "professional services"],
            "government": ["government", "public sector", "federal", "state agency"],
            "automotive": ["automotive", "vehicle", "car", "mobility"],
            "aerospace": ["aerospace", "aviation", "defense", "space"],
            "energy": ["energy", "oil", "gas", "renewable", "utilities"],
        }
        
        text_lower = text.lower()
        
        for industry, keywords in industries.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return industry.title()
        
        return ""
    
    def _extract_team_size(self, text: str) -> str:
        """Extract team size information."""
        patterns = [
            r"(?:team\s+of\s+|work\s+with\s+)(\d+(?:-\d+)?)\s*(?:people|engineers|developers|members)",
            r"(\d+(?:-\d+)?)\s*(?:person|member)\s+team",
            r"(?:small|medium|large)\s+team",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "small" in match.group(0).lower():
                    return "Small (2-5)"
                elif "medium" in match.group(0).lower():
                    return "Medium (6-15)"
                elif "large" in match.group(0).lower():
                    return "Large (15+)"
                else:
                    return match.group(1)
        
        return ""
    
    def _extract_tech_stack(self, text: str) -> List[str]:
        """Extract technology stack mentioned in job."""
        tech_stack = []
        
        # Look for tech stack section or general mentions
        patterns = [
            r"(?i)tech\s*stack[:\s]+([^\n]+)",
            r"(?i)technologies[:\s]+([^\n]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                techs = re.split(r'[,;|]', match.group(1))
                for tech in techs:
                    tech = tech.strip()
                    if tech and 2 < len(tech) < 30:
                        tech_stack.append(tech)
        
        # Also extract from skills
        text_lower = text.lower()
        for skill in self.TECH_SKILLS:
            if skill in text_lower and skill.title() not in tech_stack:
                tech_stack.append(skill.title() if len(skill) > 3 else skill.upper())
        
        return list(set(tech_stack))[:20]
    
    def _extract_project_types(self, text: str) -> List[str]:
        """Extract types of projects mentioned in job."""
        project_types = []
        
        project_keywords = {
            "Machine Learning": ["machine learning", "ml models", "ml pipeline", "ml systems"],
            "Data Engineering": ["data pipeline", "etl", "data warehouse", "data infrastructure"],
            "Web Development": ["web application", "web services", "frontend", "backend"],
            "Mobile Development": ["mobile app", "ios", "android", "mobile development"],
            "API Development": ["api development", "rest api", "graphql", "microservices"],
            "Cloud Infrastructure": ["cloud infrastructure", "aws", "azure", "gcp", "cloud architecture"],
            "DevOps": ["ci/cd", "deployment", "infrastructure", "devops"],
            "Data Science": ["data analysis", "analytics", "statistical", "data science"],
            "AI/NLP": ["nlp", "natural language", "llm", "generative ai", "chatbot"],
            "Computer Vision": ["computer vision", "image processing", "object detection"],
        }
        
        text_lower = text.lower()
        
        for project_type, keywords in project_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if project_type not in project_types:
                        project_types.append(project_type)
                    break
        
        return project_types
