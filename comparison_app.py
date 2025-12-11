import streamlit as st
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from langchain_docling import DoclingLoader
from langchain_core.documents import Document

# Page Configuration
st.set_page_config(
    page_title="Docling vs Standard Extraction",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px;
        color: #585858;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #e8f0fe;
        color: #1a73e8;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary location"""
    try:
        suffix = Path(uploaded_file.name).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def extract_with_pymupdf(file_path):
    """Extract text using PyMuPDF (Standard Method)"""
    start_time = time.time()
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n\n"  # Simple text extraction
        doc.close()
        end_time = time.time()
        return text.strip(), end_time - start_time
    except Exception as e:
        return f"Error: {str(e)}", 0

def extract_with_docling(file_path):
    """Extract text using Docling (Advanced Method)"""
    start_time = time.time()
    try:
        # Use DoclingLoader for consistency with test.py, or DocumentConverter directly
        # Using DocumentConverter is often cleaner for just getting markdown
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        
        end_time = time.time()
        return markdown_text, end_time - start_time
    except Exception as e:
        return f"Error: {str(e)}", 0

def main():
    st.title("📑 Document Extraction Comparison")
    st.markdown("""
    Compare how **Docling** handles document extraction versus standard tools like **PyMuPDF**.
    Upload a document (PDF) to see the difference in layout preservation, structure, and text quality.
    """)

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload Document", type=['pdf'])
        
        st.divider()
        st.info("ℹ️ **Note:** Docling uses advanced layout analysis and OCR (if needed), which may take longer but produces structured Markdown.")

    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        
        if file_path:
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            
            with st.spinner("Running extractors..."):
                # Run PyMuPDF
                pymupdf_text, pymupdf_time = extract_with_pymupdf(file_path)
                
                # Run Docling
                docling_text, docling_time = extract_with_docling(file_path)
            
            # Remove temp file
            os.unlink(file_path)

            # Display Metrics
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚡ Standard (PyMuPDF)</h3>
                    <p>Time: <b>{pymupdf_time:.2f}s</b></p>
                    <p>Length: {len(pymupdf_text)} chars</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🚀 Advanced (Docling)</h3>
                    <p>Time: <b>{docling_time:.2f}s</b></p>
                    <p>Length: {len(docling_text)} chars</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()

            # Tabs for viewing content
            tab1, tab2, tab3 = st.tabs(["⚡ Standard Output", "🚀 Docling Output", "ℹ️ Method Descriptions"])
            
            with tab1:
                st.subheader("Standard Text Extraction (PyMuPDF)")
                st.text_area("Raw Text Content", pymupdf_text, height=600)
                
            with tab2:
                st.subheader("Docling Markdown Extraction")
                # Docling often outputs Markdown, so we can render it
                st.markdown(docling_text)
                st.divider()
                with st.expander("View Raw Markdown Source"):
                    st.text_area("Raw Markdown", docling_text, height=400)
            
            with tab3:
                st.subheader("Method Comparison")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### ⚡ PyMuPDF (Standard)")
                    st.markdown("""
                    **What it does:**
                    - Reads the text layer directly from digital PDFs.
                    - Very fast execution.
                    - "Dumb" extraction: reads text left-to-right, top-to-bottom as it appears in the file stream.
                    
                    **Limitations:**
                    - **No Layout Understanding:** Can mix up columns, headers, and footers.
                    - **Tables:** Often destroys table structures (flattens them to text).
                    - **Scanned Docs:** Fails completely on images/scanned PDFs unless OCR is added separately.
                    - **Formatting:** Loses bold, italic, and heading hierarchy.
                    """)
                
                with c2:
                    st.markdown("### 🚀 Docling (Advanced)")
                    st.markdown("""
                    **What it does:**
                    - Uses AI/Machine Learning models to understand the *document layout*.
                    - Identifies headers, paragraphs, lists, and tables.
                    - **OCR Built-in:** Can read scanned documents and images.
                    - Exports semantic **Markdown**, preserving structure.
                    
                    **Advantages:**
                    - **Tables:** Preserves table structure (converts to Markdown tables).
                    - **Reading Order:** Correctly handles multi-column layouts.
                    - **RAG Ready:** Produces high-quality chunks for AI applications.
                    """)

    else:
        st.info("👋 Upload a PDF file in the sidebar to start the comparison.")

if __name__ == "__main__":
    main()
