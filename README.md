# Smart resume & job matcher

An AI-powered Resume and Job Matching System that uses embeddings, semantic search, and Generative AI reasoning to match candidates' resumes with the most relevant job opportunities.

## Features

- **Comprehensive Matching**: Multi-dimensional scoring combining semantic similarity with skills, experience, education, certifications, and project relevance
- **Deep Document Parsing**: Extracts 20+ fields from resumes including projects, publications, awards, interests, and detailed work history
- **Multiple Embedding Providers**: SentenceTransformers (local), Ollama (local), Google Vertex AI (cloud)
- **AI-Generated Explanations**: LangChain-powered detailed match explanations with experience analysis, project relevance, and certification fit
- **Interactive UI**: Streamlit web application for easy interaction
- **Jupyter Notebook Demo**: Complete workflow demonstration

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Resume Files  │     │ Job Descriptions│
│  (PDF/DOCX/TXT) │     │  (PDF/DOCX/TXT) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           Document Parsers              │
│  (Skills, Experience, Projects, Certs,  │
│   Education, Publications, Interests)   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│          Embedding Service              │
│  (SentenceTransformers/Ollama/Google)   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│     Comprehensive Similarity Matcher    │
│  (Semantic + Skills + Experience +      │
│   Education + Certs + Projects)         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      LangChain Explanation Chain        │
│   (AI-powered detailed explanations)    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            Streamlit UI                 │
└─────────────────────────────────────────┘
```

## Project Structure

```
koru_2/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Streamlit application
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── resume_parser.py       # Enhanced resume parsing
│   │   └── job_parser.py          # Enhanced job description parsing
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedding_service.py   # Multi-provider embedding service
│   │   └── similarity.py          # Comprehensive similarity matching
│   ├── chains/
│   │   ├── __init__.py
│   │   └── explanation_chain.py   # Enhanced LangChain explanations
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # Utility functions
├── notebooks/
│   └── demo.ipynb                 # Interactive demo notebook
├── data/
│   ├── sample_resumes/            # Sample resume files
│   └── sample_jobs/               # Sample job descriptions
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment configuration template
└── README.md                      # This file
```

## Quick Start

### 1. Clone and Setup

```bash
cd koru_2

# Create virtual environment
python -m venv koru
source koru/bin/activate  # On Windows: koru\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional - defaults work for local usage)
```

### 3. Run the Application

**Option A: Streamlit Web App**
```bash
streamlit run app/main.py
```

**Option B: Jupyter Notebook**
```bash
jupyter notebook notebooks/demo.ipynb
```

## Configuration

### Embedding Providers

| Provider | Setup | Best For |
|----------|-------|----------|
| **SentenceTransformers** | No setup needed | Local, fast, good quality |
| **Ollama** | Install Ollama, pull model | Local, flexible models |
| **Google AI** | Set `GOOGLE_API_KEY` | Cloud, high quality |

### LLM Providers (for explanations)

| Provider | Setup | Model |
|----------|-------|-------|
| **Ollama** | `ollama pull llama3.2` | Local inference |
| **Google** | Set `GOOGLE_API_KEY` | Gemini 2.5 Flash |

### Environment Variables

```env
# Embedding provider: sentence-transformers, ollama, or google
EMBEDDING_PROVIDER=sentence-transformers

# LLM provider: ollama or google
LLM_PROVIDER=ollama

# Ollama settings (if using)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Google AI settings (if using)
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
```

## Usage Guide

### Python API

```python
from app.parsers import ResumeParser, JobParser
from app.embeddings import EmbeddingService, SimilarityMatcher
from app.embeddings.similarity import compute_comprehensive_match
from app.chains import ExplanationChain

# Parse documents
resume_parser = ResumeParser()
job_parser = JobParser()

resume = resume_parser.parse(file_path="resume.pdf")
job = job_parser.parse(text="Job description text...")

# Generate embeddings
embedding_service = EmbeddingService(provider="sentence-transformers")
resume_embedding = embedding_service.embed_resume(resume)
job_embedding = embedding_service.embed_job(job)

# Get semantic similarity
matcher = SimilarityMatcher()
semantic_score = matcher.compute_similarity_matrix(
    resume_embedding.reshape(1, -1), 
    job_embedding.reshape(1, -1)
)[0, 0]

# Get comprehensive match (NEW)
match_analysis = compute_comprehensive_match(resume, job, semantic_score)
print(f"Overall Score: {match_analysis['overall_score']:.1%}")
print(f"Fit Level: {match_analysis['fit_level']}")
print(f"Skills Coverage: {match_analysis['skill_analysis']['coverage_percentage']:.1f}%")
print(f"Experience: {match_analysis['experience_analysis']['analysis']}")

# Generate AI explanation with full context (NEW)
chain = ExplanationChain(provider="ollama")
explanation = chain.explain_match(resume, job, semantic_score, match_analysis=match_analysis)
print(explanation.to_text())
```

### Streamlit App

1. **Upload Documents**: Upload resumes and job descriptions (PDF, DOCX, or TXT)
2. **Parse Files**: Click "Parse" to extract structured information
3. **Generate Matches**: Click "Generate Matches" to compute comprehensive similarity
4. **View Results**: See ranked matches with component scores and detailed analysis
5. **AI Explanations**: Generate detailed explanations including project and certification analysis

## 🔧 Supported File Formats

- **PDF**: Uses pdfplumber/PyPDF2 for text extraction
- **DOCX**: Uses python-docx for Word documents
- **TXT**: Plain text files with any encoding

## Extracted Information

### From Resumes
- **Contact**: Name, Email, Phone, Location, LinkedIn, GitHub
- **Skills**: Technical and soft skills (auto-detected + section parsing)
- **Work Experience**: Title, Company, Duration, Location, Description, Key Achievements
- **Projects**: Name, Description, Technologies, Highlights (GitHub stars, etc.)
- **Education**: Degree, Field, Institution, Year, GPA, Honors, Coursework
- **Certifications**: Name, Issuer, Year, Expiry, Credential ID
- **Publications**: Title, Venue, Year, Authors
- **Awards & Honors**: Recognition and achievements
- **Interests**: Personal interests and hobbies
- **Languages**: Language proficiencies
- **Calculated Fields**: Total years of experience

### From Job Descriptions
- **Basic Info**: Title, Company, Location, Job Type, Remote Type
- **Skills**: Required Skills, Preferred Skills, Tech Stack
- **Experience**: Required Years (min/max), Experience Level
- **Education**: Min Education Level, Specific Requirements
- **Certifications**: Required Certifications, Preferred Certifications
- **Role Details**: Responsibilities, Project Types, Industry
- **Compensation**: Salary Range (min/max), Benefits
- **Team Info**: Team Size, Company Description

## Comprehensive Matching

The system uses a **weighted multi-dimensional scoring** approach:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Semantic Similarity** | 30% | Embedding-based text similarity |
| **Skills Match** | 25% | Required skills coverage |
| **Experience Match** | 20% | Years and relevance analysis |
| **Education Match** | 10% | Degree level and field alignment |
| **Certifications** | 8% | Required/preferred cert coverage |
| **Projects** | 7% | Tech stack and type relevance |

### Fit Levels
- **Excellent** (≥80%): Strong match across all dimensions
- **Good** (65-79%): Solid match with minor gaps
- **Moderate** (50-64%): Partial match, notable gaps
- **Limited** (35-49%): Significant gaps present
- **Poor** (<35%): Not a good fit

## Sample Data

The `data/` directory contains sample resumes and job descriptions for testing:

**Sample Resumes:**
- `john_smith_data_scientist.txt` - 5+ years ML/Data Science experience
- `sarah_johnson_software_engineer.txt` - Full-stack developer with React/Node
- `michael_chen_marketing_analyst.txt` - Marketing analytics background

**Sample Jobs:**
- `senior_ml_engineer.txt` - ML engineering role at AI company
- `full_stack_developer.txt` - Remote full-stack position
- `data_analyst.txt` - Business intelligence role in retail

## Dependencies

Core:
- `streamlit` - Web application framework
- `langchain` - LLM orchestration
- `sentence-transformers` - Local embeddings

Parsing:
- `pdfplumber`, `PyPDF2` - PDF parsing
- `python-docx` - Word document parsing

ML/Data:
- `numpy`, `scikit-learn` - Numerical operations
- `pandas` - Data manipulation
- `pydantic` - Data validation

Optional:
- `ollama` - Local LLM inference
- `google-generativeai` - Google Gemini API
- `faiss-cpu` - Vector similarity search

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is for educational purposes. Please check individual library licenses for production use.

## Acknowledgments

- SentenceTransformers for embedding models
- LangChain for LLM orchestration
- Streamlit for the intuitive UI framework
- Ollama for local LLM inference
