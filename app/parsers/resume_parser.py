"""
Resume Parser Module

Parses resumes from PDF, DOCX, and TXT files to extract structured information
including skills, education, experience, certifications, and contact details.
"""

import re
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# PDF parsing
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# DOCX parsing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


@dataclass
class ParsedResume:
    """Structured representation of a parsed resume."""
    raw_text: str = ""
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: List[str] = field(default_factory=list)
    education: List[Dict[str, str]] = field(default_factory=list)
    experience: List[Dict[str, str]] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    summary: str = ""
    file_name: str = ""
    
    def to_text_for_embedding(self) -> str:
        """
        Generate a text representation optimized for embedding generation.
        
        Returns:
            Concatenated text of all resume sections
        """
        parts = []
        
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        
        if self.skills:
            parts.append(f"Skills: {', '.join(self.skills)}")
        
        if self.experience:
            exp_texts = []
            for exp in self.experience:
                exp_text = f"{exp.get('title', '')} at {exp.get('company', '')}"
                if exp.get('description'):
                    exp_text += f": {exp['description']}"
                exp_texts.append(exp_text)
            parts.append(f"Experience: {'; '.join(exp_texts)}")
        
        if self.education:
            edu_texts = []
            for edu in self.education:
                edu_text = f"{edu.get('degree', '')} from {edu.get('institution', '')}"
                edu_texts.append(edu_text)
            parts.append(f"Education: {'; '.join(edu_texts)}")
        
        if self.certifications:
            parts.append(f"Certifications: {', '.join(self.certifications)}")
        
        # Include raw text if structured extraction is limited
        if len(parts) < 3 and self.raw_text:
            parts.append(self.raw_text[:2000])
        
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "skills": self.skills,
            "education": self.education,
            "experience": self.experience,
            "certifications": self.certifications,
            "summary": self.summary,
            "file_name": self.file_name,
            "raw_text": self.raw_text[:500] + "..." if len(self.raw_text) > 500 else self.raw_text
        }


class ResumeParser:
    """
    Parser for extracting structured information from resume files.
    
    Supports PDF, DOCX, and TXT file formats.
    """
    
    # Common skills keywords for extraction
    COMMON_SKILLS = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust",
        "php", "swift", "kotlin", "scala", "r", "matlab", "sql", "html", "css",
        # Frameworks & Libraries
        "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "langchain", "langgraph", "streamlit",
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform",
        "ci/cd", "git", "github", "gitlab", "linux", "unix",
        # Data & AI
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "data analysis", "data engineering", "etl", "big data", "spark", "hadoop",
        "tableau", "power bi", "excel", "statistics",
        # Databases
        "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle",
        "sql server", "dynamodb", "cassandra", "neo4j",
        # Soft Skills
        "leadership", "communication", "teamwork", "problem solving", "agile", "scrum",
        "project management", "analytical", "critical thinking",
    ]
    
    # Section headers for parsing
    SECTION_PATTERNS = {
        "experience": r"(?i)(work\s*experience|professional\s*experience|employment|work\s*history|experience)",
        "education": r"(?i)(education|academic|qualifications|degrees)",
        "skills": r"(?i)(skills|technical\s*skills|competencies|expertise|technologies)",
        "certifications": r"(?i)(certifications?|certificates?|licenses?|credentials)",
        "summary": r"(?i)(summary|objective|profile|about\s*me|professional\s*summary)",
    }
    
    def __init__(self):
        """Initialize the resume parser."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check and log available parsing libraries."""
        self.available_formats = [".txt"]
        
        if PDF_AVAILABLE or PYPDF2_AVAILABLE:
            self.available_formats.append(".pdf")
        
        if DOCX_AVAILABLE:
            self.available_formats.append(".docx")
    
    def parse(self, file_path: str = None, file_content: bytes = None, 
              file_name: str = None) -> ParsedResume:
        """
        Parse a resume file and extract structured information.
        
        Args:
            file_path: Path to the resume file
            file_content: Raw file content (bytes)
            file_name: Original filename (used for format detection)
            
        Returns:
            ParsedResume object with extracted information
        """
        # Determine file extension
        if file_path:
            ext = Path(file_path).suffix.lower()
            file_name = Path(file_path).name
        elif file_name:
            ext = Path(file_name).suffix.lower()
        else:
            raise ValueError("Either file_path or file_name must be provided")
        
        # Extract raw text based on file format
        if ext == ".pdf":
            raw_text = self._parse_pdf(file_path, file_content)
        elif ext == ".docx":
            raw_text = self._parse_docx(file_path, file_content)
        elif ext == ".txt":
            raw_text = self._parse_txt(file_path, file_content)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Create parsed resume object
        resume = ParsedResume(raw_text=raw_text, file_name=file_name or "unknown")
        
        # Extract structured information
        resume.email = self._extract_email(raw_text)
        resume.phone = self._extract_phone(raw_text)
        resume.name = self._extract_name(raw_text)
        resume.skills = self._extract_skills(raw_text)
        resume.education = self._extract_education(raw_text)
        resume.experience = self._extract_experience(raw_text)
        resume.certifications = self._extract_certifications(raw_text)
        resume.summary = self._extract_summary(raw_text)
        
        return resume
    
    def _parse_pdf(self, file_path: str = None, file_content: bytes = None) -> str:
        """Extract text from PDF file."""
        text_parts = []
        
        if PDF_AVAILABLE:
            try:
                if file_path:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                elif file_content:
                    import io
                    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                
                if text_parts:
                    return "\n".join(text_parts)
            except Exception as e:
                print(f"pdfplumber failed: {e}, trying PyPDF2...")
        
        if PYPDF2_AVAILABLE:
            try:
                if file_path:
                    reader = PdfReader(file_path)
                elif file_content:
                    import io
                    reader = PdfReader(io.BytesIO(file_content))
                else:
                    raise ValueError("No file provided")
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                return "\n".join(text_parts)
            except Exception as e:
                raise RuntimeError(f"Failed to parse PDF: {e}")
        
        raise RuntimeError("No PDF parsing library available. Install pdfplumber or PyPDF2.")
    
    def _parse_docx(self, file_path: str = None, file_content: bytes = None) -> str:
        """Extract text from DOCX file."""
        if not DOCX_AVAILABLE:
            raise RuntimeError("python-docx is not installed. Run: pip install python-docx")
        
        try:
            if file_path:
                doc = Document(file_path)
            elif file_content:
                import io
                doc = Document(io.BytesIO(file_content))
            else:
                raise ValueError("No file provided")
            
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            raise RuntimeError(f"Failed to parse DOCX: {e}")
    
    def _parse_txt(self, file_path: str = None, file_content: bytes = None) -> str:
        """Extract text from TXT file."""
        try:
            if file_path:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif file_content:
                return file_content.decode('utf-8', errors='ignore')
            else:
                raise ValueError("No file provided")
        except Exception as e:
            raise RuntimeError(f"Failed to parse TXT: {e}")
    
    def _extract_email(self, text: str) -> str:
        """Extract email address from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else ""
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from text."""
        phone_patterns = [
            r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\+?[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}',
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""
    
    def _extract_name(self, text: str) -> str:
        """Extract candidate name (usually first line or before email)."""
        lines = text.strip().split('\n')
        
        # Name is typically in the first few lines
        for line in lines[:5]:
            line = line.strip()
            # Skip lines that look like headers or contain special characters
            if line and len(line) < 50:
                # Skip if it contains email or phone
                if '@' in line or re.search(r'\d{3}[-.\s]?\d{3}', line):
                    continue
                # Skip common headers
                if any(header in line.lower() for header in ['resume', 'cv', 'curriculum']):
                    continue
                # Return first valid-looking name line
                if re.match(r'^[A-Za-z\s\-\.]+$', line) and len(line) > 2:
                    return line
        
        return ""
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text."""
        text_lower = text.lower()
        found_skills = []
        
        # Find skills from predefined list
        for skill in self.COMMON_SKILLS:
            if skill.lower() in text_lower:
                # Proper case formatting
                found_skills.append(skill.title() if len(skill) > 3 else skill.upper())
        
        # Also look for skills section and extract additional items
        skills_section = self._extract_section(text, "skills")
        if skills_section:
            # Split by common delimiters
            additional = re.split(r'[,;•|\n]', skills_section)
            for item in additional:
                item = item.strip()
                if item and 2 < len(item) < 30:
                    if item.lower() not in [s.lower() for s in found_skills]:
                        found_skills.append(item)
        
        return list(set(found_skills))[:30]  # Limit to 30 skills
    
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
        next_section = None
        min_pos = len(remaining_text)
        
        for other_pattern in self.SECTION_PATTERNS.values():
            if other_pattern != pattern:
                other_match = re.search(other_pattern, remaining_text)
                if other_match and other_match.start() < min_pos:
                    min_pos = other_match.start()
        
        return remaining_text[:min_pos].strip()
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information."""
        education = []
        edu_section = self._extract_section(text, "education")
        
        if not edu_section:
            edu_section = text
        
        # Common degree patterns
        degree_patterns = [
            r"(Bachelor'?s?|B\.?S\.?|B\.?A\.?|Master'?s?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?|MBA|Associate'?s?)\s*(?:of|in)?\s*([A-Za-z\s]+)",
            r"(Bachelor|Master|Doctor|PhD|MBA)\s+(?:of\s+)?([A-Za-z\s]+)\s+(?:from|at)?\s*([A-Za-z\s]+(?:University|College|Institute))",
        ]
        
        for pattern in degree_patterns:
            matches = re.finditer(pattern, edu_section, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                edu_entry = {
                    "degree": groups[0] if groups[0] else "",
                    "field": groups[1].strip() if len(groups) > 1 and groups[1] else "",
                    "institution": groups[2].strip() if len(groups) > 2 and groups[2] else "",
                }
                if edu_entry["degree"]:
                    education.append(edu_entry)
        
        return education[:5]  # Limit to 5 entries
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience information."""
        experience = []
        exp_section = self._extract_section(text, "experience")
        
        if not exp_section:
            exp_section = text
        
        # Look for job title patterns
        # Pattern: Title at/- Company or Company - Title
        job_patterns = [
            r"([A-Za-z\s]+(?:Engineer|Developer|Manager|Analyst|Designer|Scientist|Lead|Director|Specialist|Consultant))\s*(?:at|@|-|–)\s*([A-Za-z0-9\s&.,]+)",
            r"([A-Za-z0-9\s&.,]+)\s*(?:-|–)\s*([A-Za-z\s]+(?:Engineer|Developer|Manager|Analyst|Designer|Scientist|Lead|Director|Specialist|Consultant))",
        ]
        
        lines = exp_section.split('\n')
        current_job = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in job_patterns:
                match = re.search(pattern, line)
                if match:
                    if current_job:
                        experience.append(current_job)
                    
                    groups = match.groups()
                    current_job = {
                        "title": groups[0].strip(),
                        "company": groups[1].strip() if len(groups) > 1 else "",
                        "description": "",
                    }
                    break
            else:
                # Add to current job description
                if current_job and len(line) > 10:
                    if current_job["description"]:
                        current_job["description"] += " " + line
                    else:
                        current_job["description"] = line
        
        if current_job:
            experience.append(current_job)
        
        return experience[:10]  # Limit to 10 entries
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications."""
        certifications = []
        cert_section = self._extract_section(text, "certifications")
        
        # Common certification patterns
        cert_patterns = [
            r"(AWS\s+[A-Za-z\s]+(?:Certified|Certification))",
            r"(Azure\s+[A-Za-z\s]+(?:Certified|Certification))",
            r"(Google\s+[A-Za-z\s]+(?:Certified|Certification))",
            r"(PMP|CISSP|CPA|CFA|CCNA|CCNP|CompTIA\s+[A-Za-z+]+)",
            r"(Certified\s+[A-Za-z\s]+)",
        ]
        
        search_text = cert_section if cert_section else text
        
        for pattern in cert_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                cert = match.group(1).strip()
                if cert and cert not in certifications:
                    certifications.append(cert)
        
        return certifications[:10]
    
    def _extract_summary(self, text: str) -> str:
        """Extract professional summary or objective."""
        summary_section = self._extract_section(text, "summary")
        
        if summary_section:
            # Take first 500 characters
            return summary_section[:500].strip()
        
        # If no explicit summary, use first paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50 and not any(header in para.lower() for header in ['email', 'phone', 'address']):
                return para[:500]
        
        return ""
