# Smart Resume & Job Matcher 🎯

An AI-powered Resume and Job Matching System that uses embeddings, semantic search, and Generative AI reasoning to match candidates' resumes with the most relevant job opportunities.

## 🌟 Features

- **Semantic Matching**: Goes beyond keyword matching using language understanding models
- **Document Parsing**: Extracts structured information from PDF, DOCX, and TXT files
- **Multiple Embedding Providers**: SentenceTransformers (local), Ollama (local), Google Vertex AI (cloud)
- **AI-Generated Explanations**: LangChain-powered match explanations with reasoning
- **Interactive UI**: Streamlit web application for easy interaction
- **Jupyter Notebook Demo**: Complete workflow demonstration

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Resume Files  │     │ Job Descriptions│
│  (PDF/DOCX/TXT) │     │  (PDF/DOCX/TXT) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           Document Parsers              │
│   (Extract skills, experience, etc.)    │
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
│        Similarity Matcher               │
│      (Cosine Similarity Ranking)        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      LangChain Explanation Chain        │
│   (AI-powered match explanations)       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│          Streamlit UI / API             │
└─────────────────────────────────────────┘
```

## 📁 Project Structure

```
koru_2/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Streamlit application
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── resume_parser.py       # Resume parsing (PDF, DOCX, TXT)
│   │   └── job_parser.py          # Job description parsing
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedding_service.py   # Multi-provider embedding service
│   │   └── similarity.py          # Semantic similarity matching
│   ├── chains/
│   │   ├── __init__.py
│   │   └── explanation_chain.py   # LangChain match explanations
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

## 🚀 Quick Start

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

## ⚙️ Configuration

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
| **Google** | Set `GOOGLE_API_KEY` | Gemini 1.5 Flash |

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
GOOGLE_MODEL=gemini-1.5-flash
```

## 📖 Usage Guide

### Streamlit App

1. **Upload Documents**: Upload resumes and job descriptions (PDF, DOCX, or TXT)
2. **Parse Files**: Click "Parse" to extract structured information
3. **Generate Matches**: Click "Generate Matches" to compute semantic similarity
4. **View Results**: See ranked matches with scores and skill overlap
5. **AI Explanations**: Generate detailed explanations for each match

### Python API

```python
from app.parsers import ResumeParser, JobParser
from app.embeddings import EmbeddingService, SimilarityMatcher
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

# Match
matcher = SimilarityMatcher()
similarity = matcher.compute_similarity_matrix(
    resume_embedding.reshape(1, -1), 
    job_embedding.reshape(1, -1)
)[0, 0]

print(f"Match Score: {similarity:.1%}")

# Generate explanation (requires Ollama or Google API)
chain = ExplanationChain(provider="ollama")
explanation = chain.explain_match(resume, job, similarity)
print(explanation.to_text())
```

## 🔧 Supported File Formats

- **PDF**: Uses pdfplumber/PyPDF2 for text extraction
- **DOCX**: Uses python-docx for Word documents
- **TXT**: Plain text files with any encoding

## 📊 Extracted Information

### From Resumes
- Name, Email, Phone
- Skills (technical and soft)
- Work Experience (title, company, description)
- Education (degree, institution)
- Certifications
- Professional Summary

### From Job Descriptions
- Job Title, Company, Location
- Required Skills
- Preferred Skills
- Experience Requirements
- Education Requirements
- Responsibilities
- Benefits
- Salary Range

## 🧪 Sample Data

The `data/` directory contains sample resumes and job descriptions for testing:

**Sample Resumes:**
- `john_smith_data_scientist.txt` - 5+ years ML/Data Science experience
- `sarah_johnson_software_engineer.txt` - Full-stack developer with React/Node
- `michael_chen_marketing_analyst.txt` - Marketing analytics background

**Sample Jobs:**
- `senior_ml_engineer.txt` - ML engineering role at AI company
- `full_stack_developer.txt` - Remote full-stack position
- `data_analyst.txt` - Business intelligence role in retail

## 📚 Dependencies

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

Optional:
- `ollama` - Local LLM inference
- `google-generativeai` - Google Gemini API
- `faiss-cpu` - Vector similarity search

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please check individual library licenses for production use.

## 🙏 Acknowledgments

- SentenceTransformers for embedding models
- LangChain for LLM orchestration
- Streamlit for the intuitive UI framework
- Ollama for local LLM inference
