import streamlit as st
import os
import re
from urllib.parse import unquote
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import html

# Page config
st.set_page_config(
    page_title="Articles Research Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for black and white theme
st.markdown("""
<style>
    .main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #ffffff !important;
    margin-bottom: 2rem;
    padding: 1.5rem;
    border-radius: 8px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff !important;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #ddd;
        padding-bottom: 0.5rem;
    }

    .answer-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #212529;
    }

    .copy-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 5px 10px;
        cursor: pointer;
        font-size: 0.8rem;
        color: #666;
    }

    .copy-btn:hover {
        background: #f8f9fa;
        border-color: #999;
    }

    .stButton > button {
        background: #000;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        width: 100%;
        margin: 0.5rem 0;
    }

    .stButton > button:hover {
        background: #333;
    }

    .sidebar .stButton > button {
        width: 100%;
        margin: 0.3rem 0;
    }

    .source-item {
        background: #f8f9fa;
        border-left: 3px solid #000;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
        min-height: 40px;
        display: flex;
        align-items: center;
    }

    .source-item a {
        color: #0066cc;
        text-decoration: none;
        word-break: break-all;
    }

    .source-item a:hover {
        text-decoration: underline;
    }

    .progress-container {
        margin: 1rem 0;
    }

    .status-text {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }

    .main-content {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    .sidebar-section {
        margin-bottom: 1.5rem;
    }

    .url-preview {
        font-size: 0.8rem;
        color: #666;
        background: #f8f9fa;
        padding: 0.3rem 0.5rem;
        border-radius: 4px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to clean HTML tags from text
def clean_html_tags(text):
    """Remove HTML tags and clean up text"""
    if not text:
        return text
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Replace common HTML entities
    clean_text = html.unescape(clean_text)
    
    # Remove extra whitespace and newlines
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Remove any remaining HTML-like artifacts
    clean_text = re.sub(r'&\w+;', '', clean_text)
    
    return clean_text.strip()

# Function to clean and decode URLs
def clean_and_decode_url(url):
    """Clean and decode URL from HTML entities and URL encoding"""
    if not url:
        return url
    
    # First, unescape HTML entities
    url = html.unescape(url)
    
    # Then, decode URL encoding
    url = unquote(url)
    
    # Remove trailing slash if it exists (except for root domains)
    if url.endswith('/') and url.count('/') > 2:
        url = url.rstrip('/')
    
    return url.strip()

# NEW: Function to extract and clean sources
def extract_and_clean_sources(sources_text):
    """Extract individual sources from the sources text and remove duplicates"""
    if not sources_text or sources_text.strip() == "":
        return []
    
    # Split by various delimiters that might separate sources
    potential_sources = re.split(r'[,\n\r\t;]', sources_text)
    
    valid_sources = []
    seen_sources = set()
    
    for source in potential_sources:
        source = source.strip()
        
        # Skip empty sources or common non-source indicators
        if not source or source.lower() in ['n/a', 'none', 'null', '']:
            continue
            
        # Clean and decode the source
        cleaned_source = clean_and_decode_url(source)
        
        # Check if it's a valid URL
        if cleaned_source.startswith(('http://', 'https://')):
            # Normalize URL for duplicate checking (remove fragments and query params for comparison)
            normalized_url = re.sub(r'[#?].*$', '', cleaned_source)
            
            # Only add if we haven't seen this URL before
            if normalized_url not in seen_sources:
                seen_sources.add(normalized_url)
                valid_sources.append(cleaned_source)
        else:
            # For non-URL sources, check for duplicates differently
            if cleaned_source not in seen_sources:
                seen_sources.add(cleaned_source)
                valid_sources.append(cleaned_source)
    
    return valid_sources

# Initialize session state
if 'urls' not in st.session_state:
    st.session_state.urls = ['']
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'vector_db_ready' not in st.session_state:
    st.session_state.vector_db_ready = False
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Adding API key for LLM call
os.environ["GOOGLE_API_KEY"] = st.secrets["api_key"]

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Main header
st.markdown('<h1 class="main-header">Articles Research Tool</h1>', unsafe_allow_html=True)

# Sidebar for URL management
st.sidebar.markdown("## URL Management")

def add_url():
    st.session_state.urls.append('')

def remove_url(index):
    if len(st.session_state.urls) > 1:
        st.session_state.urls.pop(index)

st.sidebar.markdown("### Add Article URLs")
for i, url in enumerate(st.session_state.urls):
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        st.session_state.urls[i] = st.text_input(
            f"URL {i+1}",
            value=url,
            key=f"url_{i}",
            placeholder="https://example.com/article",
            label_visibility="collapsed"
        )
    with col2:
        if len(st.session_state.urls) > 1:
            if st.button("×", key=f"remove_{i}", help="Remove this URL"):
                remove_url(i)
                st.rerun()

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("+ Add URL", key="add_url"):
        add_url()
        st.rerun()

with col2:
    if st.button("Clear All", key="clear_all"):
        st.session_state.urls = ['']
        st.session_state.vector_db = None
        st.session_state.vector_db_ready = False
        st.rerun()

valid_urls = [url.strip() for url in st.session_state.urls if url.strip()]

if valid_urls:
    st.sidebar.markdown("### Current URLs")
    for i, url in enumerate(valid_urls, 1):
        st.sidebar.markdown(f'<div class="url-preview">{i}. {url[:40]}{"..." if len(url) > 40 else ""}</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
process_clicked = st.sidebar.button("Process URLs", disabled=len(valid_urls) == 0)

if process_clicked:
    st.session_state.processing = True
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.markdown('<div class="status-text">Loading data from URLs...</div>', unsafe_allow_html=True)
        progress_bar.progress(20)
        loader = UnstructuredURLLoader(urls=valid_urls)
        data = loader.load()

        if not data:
            st.error("No data could be loaded from the provided URLs. Please check if the URLs are valid and accessible.")
            st.stop()

        status_text.markdown('<div class="status-text">Splitting text into chunks...</div>', unsafe_allow_html=True)
        progress_bar.progress(40)

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "."],
            chunk_size=600,
            chunk_overlap=50
        )
        docs = splitter.split_documents(data)

        status_text.markdown('<div class="status-text">Creating embeddings and building vector database...</div>', unsafe_allow_html=True)
        progress_bar.progress(70)

        embedder = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        vector_db = FAISS.from_documents(docs, embedding=embedder)

        status_text.markdown('<div class="status-text">Storing vector database in session...</div>', unsafe_allow_html=True)
        progress_bar.progress(90)

        # Store vector database in session state instead of pickle
        st.session_state.vector_db = vector_db

        progress_bar.progress(100)
        st.session_state.vector_db_ready = True
        st.session_state.processing = False
        st.success("Processing complete! You can now ask questions about the content.")

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        st.session_state.processing = False

# Check if vector database is ready from session state
if st.session_state.vector_db is not None:
    st.session_state.vector_db_ready = True

if st.session_state.vector_db_ready:
    st.markdown("---")
    st.markdown('<h2 class="sub-header">Ask Questions</h2>', unsafe_allow_html=True)
    query = st.text_area(
        "Enter your question:",
        placeholder="What is the main topic discussed in these articles?",
        key="query_input",
        height=100
    )
    search_clicked = st.button("Send Query", key="search_button", type="primary")

    if search_clicked and query:
        with st.spinner("Searching for relevant information..."):
            try:
                # Use vector database from session state
                vector_db = st.session_state.vector_db
                
                if vector_db is None:
                    st.error("Vector database not found. Please process URLs first.")
                    st.stop()

                retriever = vector_db.as_retriever(search_kwargs={"k": 3})
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
                result = chain.invoke({'question': query})

                st.markdown("---")
                st.markdown("### Answer")

                # Clean the answer text from HTML tags
                raw_answer = result["answer"]
                cleaned_answer = clean_html_tags(raw_answer)
                
                # Display answer without copy functionality
                st.markdown(f"""
                <div class="answer-card">
                    <div>{html.escape(cleaned_answer)}</div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")

    elif search_clicked and not query:
        st.warning("Please enter a question before searching!")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem; font-size: 0.9rem;">
    Powered by LangChain, HuggingFace, and Together AI
</div>
""", unsafe_allow_html=True)
