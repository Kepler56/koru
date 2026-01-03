"""
Smart Resume and Job Matcher - Streamlit Application

A web application for matching resumes with job descriptions using
semantic embeddings and AI-powered explanations.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Import application modules
from app.parsers import ResumeParser, JobParser
from app.embeddings import EmbeddingService, SimilarityMatcher
from app.embeddings.similarity import compute_skill_overlap
from app.chains import ExplanationChain
from app.utils.helpers import load_config, format_score

# Page configuration
st.set_page_config(
    page_title="Smart Resume & Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .match-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
    .skill-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .missing-skill-tag {
        background-color: #ffebee;
        color: #c62828;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'resumes' not in st.session_state:
        st.session_state.resumes = []
    if 'jobs' not in st.session_state:
        st.session_state.jobs = []
    if 'resume_embeddings' not in st.session_state:
        st.session_state.resume_embeddings = None
    if 'job_embeddings' not in st.session_state:
        st.session_state.job_embeddings = None
    if 'matches' not in st.session_state:
        st.session_state.matches = None
    if 'config' not in st.session_state:
        st.session_state.config = load_config()


@st.cache_resource
def get_embedding_service(provider: str):
    """Get cached embedding service instance."""
    return EmbeddingService(provider=provider)


@st.cache_resource
def get_explanation_chain(provider: str):
    """Get cached explanation chain instance."""
    try:
        return ExplanationChain(provider=provider)
    except Exception as e:
        st.warning(f"Could not initialize explanation chain: {e}")
        return None


def get_score_class(score: float) -> str:
    """Get CSS class based on score."""
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    return "score-low"


def display_skills(skills: list, title: str, is_missing: bool = False):
    """Display skills as tags."""
    if not skills:
        return
    
    tag_class = "missing-skill-tag" if is_missing else ""
    st.write(f"**{title}:**")
    
    tags_html = ""
    for skill in skills[:15]:
        tags_html += f'<span class="skill-tag {tag_class}">{skill}</span>'
    
    st.markdown(tags_html, unsafe_allow_html=True)


def parse_uploaded_files(uploaded_files, parser, file_type: str) -> list:
    """Parse uploaded files and return list of parsed objects."""
    parsed = []
    
    for file in uploaded_files:
        try:
            content = file.read()
            file.seek(0)  # Reset file pointer
            
            if file_type == "resume":
                result = parser.parse(
                    file_content=content,
                    file_name=file.name
                )
            else:
                result = parser.parse(
                    file_content=content,
                    file_name=file.name
                )
            
            parsed.append(result)
            st.success(f"✅ Parsed: {file.name}")
            
        except Exception as e:
            st.error(f"❌ Error parsing {file.name}: {str(e)}")
    
    return parsed


def main():
    """Main application entry point."""
    init_session_state()
    config = st.session_state.config
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Smart Resume & Job Matcher</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "AI-powered matching using semantic embeddings and intelligent reasoning"
        "</p>",
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Embedding provider selection
        embedding_provider = st.selectbox(
            "Embedding Provider",
            ["sentence-transformers", "ollama", "google"],
            index=0,
            help="Select the embedding model provider"
        )
        
        # LLM provider selection
        llm_provider = st.selectbox(
            "LLM Provider (for explanations)",
            ["ollama", "google"],
            index=0,
            help="Select the LLM for generating match explanations"
        )
        
        # Matching settings
        st.subheader("Matching Settings")
        top_k = st.slider("Top K Matches", 1, 10, config.get('top_k_matches', 5))
        threshold = st.slider(
            "Similarity Threshold", 
            0.0, 1.0, 
            config.get('similarity_threshold', 0.3),
            0.05
        )
        
        # API Keys (if using Google)
        if embedding_provider == "google" or llm_provider == "google":
            st.subheader("API Configuration")
            google_api_key = st.text_input(
                "Google API Key",
                value=os.getenv("GOOGLE_API_KEY", ""),
                type="password"
            )
            if google_api_key:
                os.environ["GOOGLE_API_KEY"] = google_api_key
        
        st.divider()
        
        # Clear data button
        if st.button("🗑️ Clear All Data"):
            st.session_state.resumes = []
            st.session_state.jobs = []
            st.session_state.resume_embeddings = None
            st.session_state.job_embeddings = None
            st.session_state.matches = None
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Upload Documents",
        "🔍 Match Results",
        "📊 Analysis",
        "ℹ️ About"
    ])
    
    # Tab 1: Upload Documents
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Upload Resumes")
            resume_files = st.file_uploader(
                "Upload resume files (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="resume_uploader"
            )
            
            if resume_files:
                if st.button("Parse Resumes", key="parse_resumes"):
                    with st.spinner("Parsing resumes..."):
                        parser = ResumeParser()
                        st.session_state.resumes = parse_uploaded_files(
                            resume_files, parser, "resume"
                        )
            
            # Show parsed resumes
            if st.session_state.resumes:
                st.write(f"**{len(st.session_state.resumes)} resume(s) loaded**")
                for i, resume in enumerate(st.session_state.resumes):
                    with st.expander(f"Resume: {resume.file_name}"):
                        st.write(f"**Name:** {resume.name or 'N/A'}")
                        st.write(f"**Email:** {resume.email or 'N/A'}")
                        display_skills(resume.skills, "Skills")
                        if resume.experience:
                            st.write(f"**Experience entries:** {len(resume.experience)}")
                        if resume.education:
                            st.write(f"**Education entries:** {len(resume.education)}")
        
        with col2:
            st.subheader("💼 Upload Job Descriptions")
            job_files = st.file_uploader(
                "Upload job description files (PDF, DOCX, TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="job_uploader"
            )
            
            # Manual job entry
            st.write("**Or enter job description manually:**")
            manual_job = st.text_area(
                "Job Description",
                height=150,
                placeholder="Paste job description here..."
            )
            
            if job_files or manual_job:
                if st.button("Parse Jobs", key="parse_jobs"):
                    with st.spinner("Parsing job descriptions..."):
                        parser = JobParser()
                        parsed_jobs = []
                        
                        if job_files:
                            parsed_jobs.extend(
                                parse_uploaded_files(job_files, parser, "job")
                            )
                        
                        if manual_job:
                            try:
                                job = parser.parse(text=manual_job)
                                parsed_jobs.append(job)
                                st.success("✅ Parsed manual job entry")
                            except Exception as e:
                                st.error(f"Error: {e}")
                        
                        st.session_state.jobs = parsed_jobs
            
            # Show parsed jobs
            if st.session_state.jobs:
                st.write(f"**{len(st.session_state.jobs)} job(s) loaded**")
                for i, job in enumerate(st.session_state.jobs):
                    with st.expander(f"Job: {job.title or job.file_name}"):
                        st.write(f"**Title:** {job.title or 'N/A'}")
                        st.write(f"**Company:** {job.company or 'N/A'}")
                        st.write(f"**Location:** {job.location or 'N/A'}")
                        display_skills(job.required_skills, "Required Skills")
                        display_skills(job.preferred_skills, "Preferred Skills")
                        if job.required_experience:
                            st.write(f"**Experience Required:** {job.required_experience}")
        
        # Generate embeddings and match
        st.divider()
        
        if st.session_state.resumes and st.session_state.jobs:
            if st.button("🚀 Generate Matches", type="primary"):
                with st.spinner("Generating embeddings and matching..."):
                    try:
                        # Initialize embedding service
                        embedding_service = get_embedding_service(embedding_provider)
                        
                        # Generate embeddings
                        st.info("Generating resume embeddings...")
                        resume_embeddings = embedding_service.embed_documents(
                            st.session_state.resumes
                        )
                        st.session_state.resume_embeddings = resume_embeddings
                        
                        st.info("Generating job embeddings...")
                        job_embeddings = embedding_service.embed_documents(
                            st.session_state.jobs
                        )
                        st.session_state.job_embeddings = job_embeddings
                        
                        # Match
                        st.info("Computing matches...")
                        matcher = SimilarityMatcher()
                        matches = matcher.match_resumes_to_jobs(
                            resume_embeddings,
                            job_embeddings,
                            st.session_state.resumes,
                            st.session_state.jobs,
                            top_k=top_k,
                            threshold=threshold
                        )
                        st.session_state.matches = matches
                        
                        st.success("✅ Matching complete! Go to 'Match Results' tab.")
                        
                    except Exception as e:
                        st.error(f"Error during matching: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("👆 Upload resumes and job descriptions to get started")
    
    # Tab 2: Match Results
    with tab2:
        if st.session_state.matches:
            st.subheader("🎯 Match Results")
            
            # Initialize explanation chain if available
            explanation_chain = get_explanation_chain(llm_provider)
            
            for resume_idx, resume_matches in enumerate(st.session_state.matches):
                resume = st.session_state.resumes[resume_idx]
                resume_name = resume.name or resume.file_name
                
                st.markdown(f"### 📄 {resume_name}")
                
                if not resume_matches:
                    st.warning("No matches found above threshold")
                    continue
                
                for match in resume_matches:
                    job = st.session_state.jobs[match.job_index]
                    score = match.similarity_score
                    score_class = get_score_class(score)
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="match-card">
                            <h4>💼 {job.title or job.file_name}</h4>
                            <p><strong>Company:</strong> {job.company or 'N/A'}</p>
                            <p><strong>Match Score:</strong> <span class="{score_class}">{format_score(score)}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Skill overlap analysis
                        skill_overlap = compute_skill_overlap(
                            resume.skills,
                            job.required_skills
                        )
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            display_skills(skill_overlap['matching_skills'][:10], "✅ Matching Skills")
                        with col2:
                            display_skills(skill_overlap['missing_skills'][:10], "❌ Missing Skills", is_missing=True)
                        with col3:
                            st.metric(
                                "Skill Coverage",
                                f"{skill_overlap['coverage_percentage']:.0f}%",
                                delta=f"{skill_overlap['matched_count']}/{skill_overlap['total_required']} skills"
                            )
                        
                        # Generate explanation
                        if explanation_chain:
                            with st.expander("🤖 AI Explanation"):
                                if st.button(f"Generate Explanation", key=f"explain_{resume_idx}_{match.job_index}"):
                                    with st.spinner("Generating explanation..."):
                                        try:
                                            explanation = explanation_chain.explain_match(
                                                resume, job, score, skill_overlap
                                            )
                                            st.markdown(explanation.to_text())
                                        except Exception as e:
                                            st.error(f"Error generating explanation: {e}")
                        
                        st.divider()
        else:
            st.info("🔄 Generate matches first from the 'Upload Documents' tab")
    
    # Tab 3: Analysis
    with tab3:
        st.subheader("📊 Match Analysis")
        
        if st.session_state.matches and st.session_state.resume_embeddings is not None:
            import pandas as pd
            import numpy as np
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Resumes Analyzed", len(st.session_state.resumes))
            with col2:
                st.metric("Jobs Analyzed", len(st.session_state.jobs))
            with col3:
                all_scores = []
                for matches in st.session_state.matches:
                    for m in matches:
                        all_scores.append(m.similarity_score)
                avg_score = np.mean(all_scores) if all_scores else 0
                st.metric("Average Match Score", format_score(avg_score))
            
            # Score distribution
            st.subheader("Score Distribution")
            if all_scores:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(all_scores, bins=20, edgecolor='black', alpha=0.7, color='#1E88E5')
                ax.set_xlabel("Similarity Score")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Match Scores")
                ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
                ax.legend()
                st.pyplot(fig)
            
            # Match matrix heatmap
            st.subheader("Match Matrix")
            if st.session_state.resume_embeddings is not None and st.session_state.job_embeddings is not None:
                matcher = SimilarityMatcher()
                sim_matrix = matcher.compute_similarity_matrix(
                    st.session_state.resume_embeddings,
                    st.session_state.job_embeddings
                )
                
                resume_labels = [r.name or r.file_name[:20] for r in st.session_state.resumes]
                job_labels = [j.title or j.file_name[:20] for j in st.session_state.jobs]
                
                df = pd.DataFrame(
                    sim_matrix,
                    index=resume_labels,
                    columns=job_labels
                )
                
                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.imshow(sim_matrix, cmap='Blues', aspect='auto')
                ax.set_xticks(range(len(job_labels)))
                ax.set_yticks(range(len(resume_labels)))
                ax.set_xticklabels(job_labels, rotation=45, ha='right')
                ax.set_yticklabels(resume_labels)
                ax.set_xlabel("Jobs")
                ax.set_ylabel("Resumes")
                ax.set_title("Resume-Job Similarity Matrix")
                plt.colorbar(im, ax=ax, label='Similarity Score')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show as dataframe
                st.dataframe(df.style.background_gradient(cmap='Blues'))
        else:
            st.info("Generate matches to see analysis")
    
    # Tab 4: About
    with tab4:
        st.subheader("About Smart Resume & Job Matcher")
        
        st.markdown("""
        ### 🎯 Overview
        
        This application uses **AI-powered semantic matching** to find the best matches between
        job seekers' resumes and job opportunities. Unlike traditional keyword matching, it
        understands the **meaning and context** of skills, experience, and requirements.
        
        ### 🔧 How It Works
        
        1. **Document Parsing**: Resumes and job descriptions are parsed to extract structured
           information (skills, experience, education, requirements).
        
        2. **Embedding Generation**: Text is converted into high-dimensional vectors using
           state-of-the-art language models (SentenceTransformers, Ollama, or Google AI).
        
        3. **Semantic Matching**: Cosine similarity is computed between resume and job embeddings
           to find contextually relevant matches.
        
        4. **AI Explanations**: LangChain orchestrates LLM calls to generate human-readable
           explanations of why candidates match specific positions.
        
        ### 📚 Supported Formats
        
        - **Resumes**: PDF, DOCX, TXT
        - **Job Descriptions**: PDF, DOCX, TXT, or direct text input
        
        ### 🛠️ Technology Stack
        
        - **Frontend**: Streamlit
        - **Embeddings**: SentenceTransformers, Ollama, Google Generative AI
        - **LLM Orchestration**: LangChain
        - **Similarity**: NumPy, scikit-learn
        
        ### 📝 Configuration
        
        Set up your `.env` file based on `.env.example` to configure:
        - Embedding provider and model
        - LLM provider for explanations
        - API keys (if using cloud services)
        """)


if __name__ == "__main__":
    main()
