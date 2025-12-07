import os
import json
import streamlit as st
import sqlite3
import pickle
from pathlib import Path
from tempfile import mkdtemp
import shutil
import warnings
import base64
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image
import asyncio
import threading

# Suppress warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")

from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_milvus import Milvus

# Azure OpenAI Configuration
AZURE_CONFIG = {
    "AZURE_OPENAI_ENDPOINT": "https://oai-nasco.openai.azure.com",
    "AZURE_OPENAI_API_KEY": "YOUR_AZURE_OPENAI_API_KEY",
    "OPENAI_API_VERSION": "2025-01-01-preview",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": "gpt-4o",
    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME": "text-embedding-ada-002"
}

# Set environment variables
for key, value in AZURE_CONFIG.items():
    os.environ[key] = value

# Configuration
DATABASE_DIR = Path("./document_databases")
DATABASE_DIR.mkdir(exist_ok=True)
FILES_DIR = Path("./uploaded_files")
FILES_DIR.mkdir(exist_ok=True)

TOP_K = 3
PROMPT = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {input}\nAnswer:\n",
)

def setup_event_loop():
    """Setup event loop for async operations in Streamlit thread"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'vectorstore': None,
        'llm': None,
        'embedding': None,
        'current_database': None,
        'available_databases': [],
        'processed_files': [],
        'file_contents': {},
        'file_metadata': {},
        'top_k': TOP_K,
        'last_response': None,
        'pdf_viewer_active': False,
        'current_pdf_page': None,
        'current_pdf_file': None,
        'current_question': "",
        'chat_history': [],
        'button_counter': 0,
        'last_processed_question': ""
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_available_databases():
    """Get list of available databases"""
    databases = []
    for db_path in DATABASE_DIR.glob("*.db"):
        db_name = db_path.stem
        databases.append(db_name)
    return databases

def save_database_metadata(db_name, files_info):
    """Save metadata about the database"""
    metadata_path = DATABASE_DIR / f"{db_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(files_info, f, indent=2)

def load_database_metadata(db_name):
    """Load metadata about the database"""
    metadata_path = DATABASE_DIR / f"{db_name}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def initialize_azure_models():
    """Initialize Azure OpenAI models"""
    try:
        embedding = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_CONFIG["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
            openai_api_version=AZURE_CONFIG["OPENAI_API_VERSION"],
        )
        
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            openai_api_version=AZURE_CONFIG["OPENAI_API_VERSION"],
            temperature=0,
        )
        
        return embedding, llm
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI models: {str(e)}")
        return None, None

def save_file_permanently(uploaded_file, db_name):
    """Save uploaded file permanently for later access"""
    try:
        db_files_dir = FILES_DIR / db_name
        db_files_dir.mkdir(exist_ok=True)
        
        file_path = db_files_dir / uploaded_file.name
        file_content = uploaded_file.getbuffer()
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path), file_content
    except Exception as e:
        st.error(f"Error saving file permanently: {str(e)}")
        return None, None

def load_saved_files(db_name):
    """Load previously saved files for a database"""
    db_files_dir = FILES_DIR / db_name
    file_contents = {}
    
    if not db_files_dir.exists():
        st.warning(f"Files directory {db_files_dir} does not exist")
        return file_contents
    
    for file_path in db_files_dir.iterdir():
        if file_path.is_file():
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    file_contents[file_path.name] = content
            except Exception as e:
                st.error(f"Could not load file {file_path.name}: {str(e)}")
    
    return file_contents

def process_documents(file_paths, export_type):
    """Process documents using Docling's built-in capabilities"""
    try:
        # Initialize Docling loader with chunker if needed
        if export_type == ExportType.DOC_CHUNKS:
            chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
            loader = DoclingLoader(
                file_path=file_paths,
                export_type=export_type,
                chunker=chunker,
            )
        else:
            loader = DoclingLoader(
                file_path=file_paths,
                export_type=export_type,
            )
        
        # Load documents (Docling handles OCR for scanned PDFs automatically)
        docs = loader.load()
        
        # Process based on export type
        if export_type == ExportType.DOC_CHUNKS:
            splits = docs
        elif export_type == ExportType.MARKDOWN:
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header_1"),
                    ("##", "Header_2"),
                    ("###", "Header_3"),
                ],
            )
            splits = []
            for doc in docs:
                doc_splits = splitter.split_text(doc.page_content)
                # Preserve metadata for each split
                for split in doc_splits:
                    split.metadata.update(doc.metadata)
                splits.extend(doc_splits)
        else:
            splits = docs
        
        return splits
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None

def create_and_save_vector_store(splits, embedding, db_name):
    """Create vector store and save it persistently"""
    try:
        # Setup event loop for async operations
        setup_event_loop()
        
        db_path = DATABASE_DIR / f"{db_name}.db"
        
        # Use synchronous approach to avoid async issues
        vectorstore = Milvus.from_documents(
            documents=splits,
            embedding=embedding,
            collection_name=f"collection_{db_name}",
            connection_args={
                "uri": str(db_path),
                "token": ""  # Add empty token to force sync mode
            },
            index_params={"index_type": "FLAT", "metric_type": "COSINE"},
            drop_old=True,
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def load_vector_store(db_name, embedding):
    """Load existing vector store"""
    try:
        # Setup event loop for async operations
        setup_event_loop()
        
        db_path = DATABASE_DIR / f"{db_name}.db"
        if not db_path.exists():
            return None
        
        vectorstore = Milvus(
            embedding_function=embedding,
            collection_name=f"collection_{db_name}",
            connection_args={
                "uri": str(db_path),
                "token": ""  # Add empty token to force sync mode
            },
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def get_pdf_page_image(pdf_content, page_number):
    """Extract a specific page from PDF as image"""
    try:
        # Make sure we have bytes content
        if isinstance(pdf_content, str):
            pdf_content = pdf_content.encode('utf-8')
        
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        if page_number <= pdf_doc.page_count and page_number > 0:
            page = pdf_doc[page_number - 1]
            # Render at higher resolution for better quality
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            pdf_doc.close()
            return img_data
        else:
            pdf_doc.close()
            st.error(f"Page {page_number} does not exist. PDF has {pdf_doc.page_count} pages.")
            return None
    except Exception as e:
        st.error(f"Error extracting PDF page: {str(e)}")
        return None

def normalize_filename(filename):
    """Normalize filename for consistent matching"""
    return Path(filename).name

def display_pdf_page(filename, page_number):
    """Display a specific PDF page"""
    # Debug: show what we're looking for and what's available
    st.info(f"Looking for PDF: '{filename}'")
    
    if st.session_state.file_contents:
        with st.expander("🔍 Debug: Available files in memory"):
            st.write("Files in file_contents:")
            for file_key in st.session_state.file_contents.keys():
                st.write(f"- '{file_key}'")
    
    if filename in st.session_state.file_contents:
        pdf_content = st.session_state.file_contents[filename]
        img_data = get_pdf_page_image(pdf_content, page_number)
        
        if img_data:
            st.markdown(f"### 📄 {filename} - Page {page_number}")
            
            # Display the image
            st.image(
                img_data,
                caption=f"Page {page_number} from {filename}",
                use_container_width=True
            )
            
            # Close button
            if st.button("❌ Close Page Viewer", key=f"close_viewer_{filename}_{page_number}"):
                st.session_state.pdf_viewer_active = False
                st.session_state.current_pdf_page = None
                st.session_state.current_pdf_file = None
                st.rerun()
        else:
            st.error("Could not load the PDF page.")
    else:
        st.error(f"PDF file '{filename}' not found in database.")
        
        # Show more debug info
        if st.session_state.file_contents:
            st.write("Available files:")
            for file_key in st.session_state.file_contents.keys():
                st.write(f"- {file_key}")
        else:
            st.error("No files loaded in memory. This usually means:")
            st.write("1. The database was created before file saving was implemented")
            st.write("2. The files were not saved properly during database creation")
            st.write("3. Please re-upload and process your documents")

def display_source_references(context_docs):
    """Display source references with clickable links to view PDF pages"""
    if not context_docs:
        return
    
    st.subheader("📚 Source References")
    
    # Group documents by source file
    sources_by_file = {}
    for i, doc in enumerate(context_docs):
        source = doc.metadata.get('source', 'Unknown')
        filename = normalize_filename(source) if source != 'Unknown' else 'Unknown'
        page = doc.metadata.get('page', doc.metadata.get('page_number', 1))
        
        if filename not in sources_by_file:
            sources_by_file[filename] = []
        sources_by_file[filename].append({
            'page': page,
            'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            'doc_index': i
        })
    
    # Display sources
    for filename, pages_info in sources_by_file.items():
        with st.expander(f"📄 {filename} ({len(pages_info)} references)", expanded=True):
            for page_info in pages_info:
                page = page_info['page']
                preview = page_info['content_preview']
                doc_index = page_info['doc_index']
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Page {page}:**")
                    st.write(f"_{preview}_")
                
                with col2:
                    # Only show view button for PDF files
                    if filename.lower().endswith('.pdf'):
                        view_key = f"view_page_{filename}_{page}_{doc_index}_{st.session_state.button_counter}"
                        st.session_state.button_counter += 1
                        if st.button("👁️ View", key=view_key, help=f"View page {page}"):
                            st.session_state.pdf_viewer_active = True
                            st.session_state.current_pdf_file = filename
                            st.session_state.current_pdf_page = page
                            st.rerun()
                    else:
                        st.write("_(Non-PDF)_")
                
                st.divider()

def display_chat_history():
    """Display the chat history in a conversational format"""
    if not st.session_state.chat_history:
        st.info("💬 Start a conversation by asking a question about your documents below.")
        return
    
    st.subheader("💬 Conversation History")
    
    # Display chat messages
    for i, chat_item in enumerate(st.session_state.chat_history):
        # User question
        with st.chat_message("user", avatar="🧑‍💼"):
            st.write(chat_item["question"])
        
        # Assistant response
        with st.chat_message("assistant", avatar="🤖"):
            st.write(chat_item["answer"])
            
            # Show sources in an expander for each response
            if chat_item.get("context") and len(chat_item["context"]) > 0:
                with st.expander(f"📚 View {len(chat_item['context'])} source references", expanded=False):
                    display_source_references_compact(chat_item["context"], i)

def display_source_references_compact(context_docs, chat_index):
    """Display source references in a more compact format for chat history"""
    if not context_docs:
        return
    
    # Group documents by source file
    sources_by_file = {}
    for doc_idx, doc in enumerate(context_docs):
        source = doc.metadata.get('source', 'Unknown')
        filename = normalize_filename(source) if source != 'Unknown' else 'Unknown'
        page = doc.metadata.get('page', doc.metadata.get('page_number', 1))
        
        if filename not in sources_by_file:
            sources_by_file[filename] = []
        sources_by_file[filename].append({
            'page': page,
            'content_preview': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
            'doc_index': doc_idx
        })
    
    # Display sources in a compact format
    for file_idx, (filename, pages_info) in enumerate(sources_by_file.items()):
        st.markdown(f"**📄 {filename}** ({len(pages_info)} references)")
        
        for page_info in pages_info:
            page = page_info['page']
            preview = page_info['content_preview']
            doc_index = page_info['doc_index']
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.caption(f"Page {page}: _{preview}_")
            
            with col2:
                if filename.lower().endswith('.pdf'):
                    st.session_state.button_counter += 1
                    view_key = f"compact_view_{st.session_state.button_counter}"
                    if st.button("👁️", key=view_key, help=f"View page {page}", use_container_width=True):
                        st.session_state.pdf_viewer_active = True
                        st.session_state.current_pdf_file = filename
                        st.session_state.current_pdf_page = page
                        st.rerun()

def add_to_chat_history(question, answer, context_docs=None):
    """Add a new chat exchange to the history"""
    chat_item = {
        "question": question,
        "answer": answer,
        "context": context_docs,
        "timestamp": len(st.session_state.chat_history)  # Use index as timestamp
    }
    st.session_state.chat_history.append(chat_item)

def clear_chat_history():
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.session_state.current_question = ""
    st.session_state.last_response = None

def render_sidebar():
    """Render the professional sidebar"""
    with st.sidebar:
        st.header("📑 Document Management")
        
        # Database selection
        st.subheader("Database Operations")
        
        # Get available databases
        available_dbs = get_available_databases()
        st.session_state.available_databases = available_dbs
        
        # Database selection dropdown
        if available_dbs:
            selected_db = st.selectbox(
                "Select Database:",
                options=["Create New..."] + available_dbs,
                help="Choose existing database or create new one"
            )
            
            if selected_db != "Create New...":
                if st.button("Load Database", type="primary", use_container_width=True):
                    load_database(selected_db)
        else:
            st.info("No existing databases found.")
            selected_db = "Create New..."
        
        # Database creation
        if selected_db == "Create New..." or not available_dbs:
            st.subheader("Create New Database")
            new_db_name = st.text_input(
                "Database Name:",
                placeholder="Enter database name",
                help="Name for the new document database"
            )
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload Documents:",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'doc', 'txt', 'md'],
                help="Support for PDF, Word, Text, and Markdown files"
            )
            
            # Export type selection
            export_type = st.selectbox(
                "Processing Mode:",
                options=[ExportType.DOC_CHUNKS, ExportType.MARKDOWN],
                format_func=lambda x: "Smart Chunks" if x == ExportType.DOC_CHUNKS else "Markdown Structure",
                help="Choose document processing method"
            )
            
            # Process button
            if uploaded_files and new_db_name and st.button("Process Documents", type="primary", use_container_width=True):
                process_new_database(new_db_name, uploaded_files, export_type)
        
        # Current database info
        if st.session_state.current_database:
            st.divider()
            st.subheader("Current Database")
            st.success(f"**Active:** {st.session_state.current_database}")
            
            if st.session_state.processed_files:
                st.write("**Documents:**")
                for i, filename in enumerate(st.session_state.processed_files, 1):
                    st.write(f"📄 {filename}")
                    
            # Show file loading status
            if st.session_state.file_contents:
                st.write(f"**Files in Memory:** {len(st.session_state.file_contents)}")
            else:
                st.error("⚠️ No files loaded in memory. PDF viewing won't work.")
            
            # Advanced settings
            st.subheader("Query Settings")
            top_k = st.slider(
                "Retrieval Count:",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                help="Number of relevant chunks to retrieve"
            )
            st.session_state.top_k = top_k

def load_database(db_name):
    """Load an existing database"""
    try:
        with st.spinner(f"Loading database '{db_name}'..."):
            # Initialize models if not already done
            if not st.session_state.embedding or not st.session_state.llm:
                embedding, llm = initialize_azure_models()
                if not embedding or not llm:
                    return
                st.session_state.embedding = embedding
                st.session_state.llm = llm
            
            # Load vector store
            vectorstore = load_vector_store(db_name, st.session_state.embedding)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.current_database = db_name
                
                # Load metadata
                metadata = load_database_metadata(db_name)
                st.session_state.processed_files = list(metadata.keys())
                st.session_state.file_metadata = metadata
                
                # Load file contents - CRITICAL for PDF viewing
                file_contents = load_saved_files(db_name)
                st.session_state.file_contents = file_contents
                
                if file_contents:
                    st.success(f"✅ Database '{db_name}' loaded successfully with {len(file_contents)} files in memory!")
                else:
                    st.warning("⚠️ Database loaded but no file contents found. PDF viewing will not work.")
                    st.info("💡 To fix this: Re-upload and process your documents to save file contents.")
                
                st.rerun()
            else:
                st.error(f"❌ Failed to load database '{db_name}'")
                
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")

def process_new_database(db_name, uploaded_files, export_type):
    """Process and create a new database"""
    try:
        with st.spinner("Processing documents and creating database..."):
            # Initialize models
            embedding, llm = initialize_azure_models()
            if not embedding or not llm:
                return
            
            # Save files permanently and keep content in memory
            file_paths = []
            file_contents = {}
            files_info = {}
            
            for uploaded_file in uploaded_files:
                temp_path, content = save_file_permanently(uploaded_file, db_name)
                if temp_path and content:
                    file_paths.append(temp_path)
                    file_contents[uploaded_file.name] = content
                    files_info[uploaded_file.name] = {
                        'size': len(content),
                        'type': uploaded_file.type,
                        'path': temp_path
                    }
            
            if file_paths:
                # Process documents using Docling's built-in OCR
                splits = process_documents(file_paths, export_type)
                
                if splits:
                    # Create and save vector store
                    vectorstore = create_and_save_vector_store(splits, embedding, db_name)
                    
                    if vectorstore:
                        # Update session state
                        st.session_state.vectorstore = vectorstore
                        st.session_state.current_database = db_name
                        st.session_state.processed_files = list(file_contents.keys())
                        st.session_state.file_contents = file_contents
                        st.session_state.file_metadata = files_info
                        st.session_state.embedding = embedding
                        st.session_state.llm = llm
                        
                        # Save metadata
                        save_database_metadata(db_name, files_info)
                        
                        st.success(f"✅ Database '{db_name}' created successfully!")
                        st.info(f"📊 Processed {len(uploaded_files)} documents into {len(splits)} chunks")
                        st.rerun()
                
    except Exception as e:
        st.error(f"Error creating database: {str(e)}")

def main():
    st.set_page_config(
        page_title="Document Intelligence Platform",
        page_icon="📑",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("📑 Document Intelligence Platform")
    st.markdown("*Advanced document processing and question answering system with Docling OCR*")
    
    # Check if PDF viewer is active
    if st.session_state.pdf_viewer_active and st.session_state.current_pdf_file and st.session_state.current_pdf_page:
        # Display PDF page viewer
        display_pdf_page(st.session_state.current_pdf_file, st.session_state.current_pdf_page)
    else:
        # Only show chat interface if we have a database loaded
        if st.session_state.vectorstore and st.session_state.current_database:
            display_chat_history()
            
            # Input section at the bottom
            st.subheader("🔍 Ask a Question")
            
            # Use form to better handle input submission
            with st.form(key='question_form'):
                question = st.text_input(
                    "Your question:",
                    placeholder="What would you like to know about your documents?",
                    help="Ask any question about your uploaded documents"
                )
                search_button = st.form_submit_button("🚀 Ask", type="primary", use_container_width=False)

            # Execute search only when form is submitted with a valid question
            if search_button and question.strip() and question != st.session_state.get('last_processed_question', ''):
                try:
                    with st.spinner("🤖 Thinking..."):
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": st.session_state.get('top_k', TOP_K)}
                        )
                        question_answer_chain = create_stuff_documents_chain(
                            st.session_state.llm, PROMPT
                        )
                        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                        
                        # Invoke the chain
                        response = rag_chain.invoke({"input": question})
                        
                        # Add to chat history
                        add_to_chat_history(
                            question=question,
                            answer=response["answer"],
                            context_docs=response.get("context", [])
                        )
                        
                        # Update last processed question to prevent re-processing
                        st.session_state.last_processed_question = question
                        
                        # Rerun to show the new message
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    
        else:
            # Welcome message
            st.info("👋 Welcome to the Document Intelligence Platform. Please upload documents or load an existing database from the sidebar to get started.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### 🧠 Smart Processing")
                st.write("Advanced document chunking with Docling's built-in OCR for scanned PDFs and images.")
            with col2:
                st.markdown("### 💾 Persistent Storage")
                st.write("Documents and indexes are saved permanently. Load any previous database instantly.")
            with col3:
                st.markdown("### 🎯 Intelligent Search")
                st.write("Natural language queries with precise source attribution and page viewing.")

if __name__ == "__main__":
    main()
