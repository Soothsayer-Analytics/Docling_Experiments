import os
import json
import streamlit as st
import sqlite3
import pickle
from pathlib import Path
import warnings
import base64
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image
import asyncio
import threading
import logging
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

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
LOGS_DIR = Path("./processing_logs")
LOGS_DIR.mkdir(exist_ok=True)

TOP_K = 3
PROMPT = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {input}\nAnswer:\n",
)

# Get number of CPU cores (limit to 4 as mentioned)
NUM_WORKERS = min(4, mp.cpu_count())

def setup_logging(db_name):
    """Setup logging for processing operations"""
    log_file = LOGS_DIR / f"{db_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger(f"doc_processor_{db_name}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler for Streamlit
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file



def standardize_document_metadata(splits):
    """Standardize metadata across all document splits to avoid Milvus schema issues"""
    if not splits:
        return splits
    
    # Collect all possible metadata keys from all documents
    all_metadata_keys = set()
    for doc in splits:
        all_metadata_keys.update(doc.metadata.keys())
    
    # Standardize each document's metadata
    for doc in splits:
        # Add missing keys with default values
        for key in all_metadata_keys:
            if key not in doc.metadata:
                if key.startswith('Header_'):
                    doc.metadata[key] = ""  # Empty string for header fields
                elif key in ['page', 'page_number']:
                    doc.metadata[key] = 1  # Default page number
                elif key == 'source':
                    doc.metadata[key] = doc.metadata.get('source', 'unknown')
                else:
                    doc.metadata[key] = ""  # Default to empty string for other fields
    
    return splits

def process_single_document(file_info, export_type):
    """Process a single document - designed to run in parallel"""
    file_path, filename = file_info
    start_time = time.time()
    
    try:
        # Initialize Docling loader for single file
        if export_type == ExportType.DOC_CHUNKS:
            chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
            loader = DoclingLoader(
                file_path=file_path,
                export_type=export_type,
                chunker=chunker,
            )
        else:
            loader = DoclingLoader(
                file_path=file_path,
                export_type=export_type,
            )
        
        # Load and process document
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
                for split in doc_splits:
                    split.metadata.update(doc.metadata)
                splits.extend(doc_splits)
        else:
            splits = docs
        
        # Ensure all splits have consistent basic metadata
        for split in splits:
            # Ensure source is set
            if 'source' not in split.metadata or not split.metadata['source']:
                split.metadata['source'] = filename
            
            # Ensure page number is set
            if 'page' not in split.metadata and 'page_number' not in split.metadata:
                split.metadata['page'] = 1
        
        processing_time = time.time() - start_time
        
        return {
            'filename': filename,
            'splits': splits,
            'processing_time': processing_time,
            'chunks_count': len(splits),
            'success': True,
            'error': None
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            'filename': filename,
            'splits': [],
            'processing_time': processing_time,
            'chunks_count': 0,
            'success': False,
            'error': str(e)
        }

def process_documents_parallel(file_paths_info, export_type, logger):
    """Process documents in parallel using multiprocessing with improved error handling"""
    logger.info(f"Starting parallel processing of {len(file_paths_info)} documents using {NUM_WORKERS} workers")
    
    all_splits = []
    processing_stats = {}
    
    # Create partial function with export_type
    process_func = partial(process_single_document, export_type=export_type)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_func, file_info): file_info[1] 
                for file_info in file_paths_info
            }
            
            completed = 0
            total_files = len(file_paths_info)
            
            # Process completed tasks
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        all_splits.extend(result['splits'])
                        logger.info(
                            f"✅ {result['filename']}: {result['chunks_count']} chunks "
                            f"in {result['processing_time']:.2f}s"
                        )
                    else:
                        logger.error(
                            f"❌ {result['filename']}: Failed in {result['processing_time']:.2f}s "
                            f"- {result['error']}"
                        )
                    
                    processing_stats[result['filename']] = {
                        'processing_time': result['processing_time'],
                        'chunks_count': result['chunks_count'],
                        'success': result['success'],
                        'error': result['error']
                    }
                    
                except Exception as e:
                    logger.error(f"❌ {filename}: Exception during processing - {str(e)}")
                    processing_stats[filename] = {
                        'processing_time': 0,
                        'chunks_count': 0,
                        'success': False,
                        'error': str(e)
                    }
                
                completed += 1
                progress_bar.progress(completed / total_files)
                status_text.text(f"Processed {completed}/{total_files} files...")
        
        # Log summary
        total_time = sum(stats['processing_time'] for stats in processing_stats.values())
        total_chunks = sum(stats['chunks_count'] for stats in processing_stats.values())
        successful_files = sum(1 for stats in processing_stats.values() if stats['success'])
        
        logger.info(f"📊 Processing Summary:")
        logger.info(f"   Total files: {total_files}")
        logger.info(f"   Successful: {successful_files}")
        logger.info(f"   Failed: {total_files - successful_files}")
        logger.info(f"   Total chunks: {total_chunks}")
        logger.info(f"   Total processing time: {total_time:.2f}s")
        logger.info(f"   Average time per file: {total_time/total_files:.2f}s" if total_files > 0 else "   Average time per file: 0s")
        
        # Log metadata information for debugging
        if all_splits:
            sample_keys = set(all_splits[0].metadata.keys()) if all_splits else set()
            logger.info(f"   Sample metadata keys: {sample_keys}")
            
            # Check for metadata consistency
            metadata_keys_sets = [set(split.metadata.keys()) for split in all_splits[:10]]  # Check first 10
            if len(set(frozenset(keys) for keys in metadata_keys_sets)) > 1:
                logger.warning("   Detected inconsistent metadata keys across documents")
            else:
                logger.info("   Metadata keys are consistent across sample documents")
        
        progress_bar.empty()
        status_text.empty()
        
        return all_splits, processing_stats
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return [], {}

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
        'pdf_viewer_active': False,
        'current_pdf_page': None,
        'current_pdf_file': None,
        'chat_history': [],
        'button_counter': 0,
        'last_processed_question': ""
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_available_databases():
    """Get list of available databases"""
    return [db_path.stem for db_path in DATABASE_DIR.glob("*.db")]

def save_database_metadata(db_name, files_info, processing_stats=None):
    """Save metadata about the database including processing stats"""
    metadata = {
        'files_info': files_info,
        'processing_stats': processing_stats,
        'created_at': datetime.now().isoformat(),
        'total_files': len(files_info),
        'num_workers_used': NUM_WORKERS
    }
    
    metadata_path = DATABASE_DIR / f"{db_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_database_metadata(db_name):
    """Load metadata about the database"""
    metadata_path = DATABASE_DIR / f"{db_name}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            # Handle both old and new metadata formats
            if 'files_info' in data:
                return data['files_info']
            else:
                return data
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
        return file_contents
    
    for file_path in db_files_dir.iterdir():
        if file_path.is_file():
            try:
                with open(file_path, 'rb') as f:
                    file_contents[file_path.name] = f.read()
            except Exception as e:
                st.error(f"Could not load file {file_path.name}: {str(e)}")
    
    return file_contents

def create_and_save_vector_store(splits, embedding, db_name, logger):
    """Create vector store and save it persistently with metadata standardization"""
    try:
        logger.info("Standardizing document metadata...")
        
        # Standardize metadata across all documents
        standardized_splits = standardize_document_metadata(splits)
        
        logger.info(f"Creating vector store with {len(standardized_splits)} standardized documents...")
        start_time = time.time()
        
        db_path = DATABASE_DIR / f"{db_name}.db"
        
        # Remove existing database file if it exists to avoid schema conflicts
        if db_path.exists():
            db_path.unlink()
            logger.info("Removed existing database file to avoid schema conflicts")
        
        vectorstore = Milvus.from_documents(
            documents=standardized_splits,
            embedding=embedding,
            collection_name=f"collection_{db_name}",
            connection_args={
                "uri": str(db_path),
                "token": ""
            },
            index_params={"index_type": "FLAT", "metric_type": "COSINE"},
            drop_old=True,
        )
        
        vector_time = time.time() - start_time
        logger.info(f"✅ Vector store created successfully in {vector_time:.2f}s")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        # Log metadata info for debugging
        if splits:
            sample_metadata_keys = set(splits[0].metadata.keys()) if splits else set()
            logger.error(f"Sample metadata keys from first document: {sample_metadata_keys}")
        return None

def load_vector_store(db_name, embedding):
    """Load existing vector store"""
    try:
        db_path = DATABASE_DIR / f"{db_name}.db"
        if not db_path.exists():
            return None
        
        vectorstore = Milvus(
            embedding_function=embedding,
            collection_name=f"collection_{db_name}",
            connection_args={
                "uri": str(db_path),
                "token": ""
            },
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def get_pdf_page_image(pdf_content, page_number):
    """Extract a specific page from PDF as image"""
    try:
        if isinstance(pdf_content, str):
            pdf_content = pdf_content.encode('utf-8')
        
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        if page_number <= pdf_doc.page_count and page_number > 0:
            page = pdf_doc[page_number - 1]
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
    if filename in st.session_state.file_contents:
        pdf_content = st.session_state.file_contents[filename]
        img_data = get_pdf_page_image(pdf_content, page_number)
        
        if img_data:
            st.markdown(f"### 📄 {filename} - Page {page_number}")
            st.image(
                img_data,
                caption=f"Page {page_number} from {filename}",
                use_container_width=True
            )
            
            if st.button("❌ Close Page Viewer", key=f"close_viewer_{filename}_{page_number}"):
                st.session_state.pdf_viewer_active = False
                st.session_state.current_pdf_page = None
                st.session_state.current_pdf_file = None
                st.rerun()
        else:
            st.error("Could not load the PDF page.")
    else:
        st.error(f"PDF file '{filename}' not found in database.")



def display_chat_history():
    """Display the chat history in a conversational format"""
    if not st.session_state.chat_history:
        st.info("💬 Start a conversation by asking a question about your documents below.")
        return
    
    st.subheader("💬 Conversation History")
    
    for i, chat_item in enumerate(st.session_state.chat_history):
        with st.chat_message("user", avatar="🧑‍💼"):
            st.write(chat_item["question"])
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(chat_item["answer"])
            
            

def add_to_chat_history(question, answer, context_docs=None):
    """Add a new chat exchange to the history"""
    chat_item = {
        "question": question,
        "answer": answer,
        "context": context_docs,
        "timestamp": len(st.session_state.chat_history)
    }
    st.session_state.chat_history.append(chat_item)

def render_sidebar():
    """Render the professional sidebar"""
    with st.sidebar:
        st.header("📑 Document Management")
        
        st.subheader("Database Operations")
        
        available_dbs = get_available_databases()
        st.session_state.available_databases = available_dbs
        
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
        
        if selected_db == "Create New..." or not available_dbs:
            st.subheader("Create New Database")
            new_db_name = st.text_input(
                "Database Name:",
                placeholder="Enter database name",
                help="Name for the new document database"
            )
            
            uploaded_files = st.file_uploader(
                "Upload Documents:",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'doc', 'txt', 'md'],
                help="Support for PDF, Word, Text, and Markdown files"
            )
            
            export_type = st.selectbox(
                "Processing Mode:",
                options=[ExportType.DOC_CHUNKS, ExportType.MARKDOWN],
                format_func=lambda x: "Smart Chunks" if x == ExportType.DOC_CHUNKS else "Markdown Structure",
                help="Choose document processing method"
            )
            
            if uploaded_files and new_db_name and st.button("Process Documents", type="primary", use_container_width=True):
                process_new_database(new_db_name, uploaded_files, export_type)
        
        if st.session_state.current_database:
            st.divider()
            st.subheader("Current Database")
            st.success(f"**Active:** {st.session_state.current_database}")
            
            if st.session_state.processed_files:
                st.write("**Documents:**")
                for filename in st.session_state.processed_files:
                    st.write(f"📄 {filename}")
            
            
            
            # Show processing logs
            if st.button("📊 View Processing Logs", use_container_width=True):
                show_processing_logs()

def show_processing_logs():
    """Display processing logs in an expandable section"""
    log_files = list(LOGS_DIR.glob("*.log"))
    
    if log_files:
        st.subheader("📊 Processing Logs")
        
        # Sort by creation time (newest first)
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for log_file in log_files[:5]:  # Show last 5 log files
            with st.expander(f"📋 {log_file.name}"):
                try:
                    with open(log_file, 'r') as f:
                        st.code(f.read(), language='text')
                except Exception as e:
                    st.error(f"Could not read log file: {str(e)}")
    else:
        st.info("No processing logs found.")

def load_database(db_name):
    """Load an existing database"""
    try:
        with st.spinner(f"Loading database '{db_name}'..."):
            if not st.session_state.embedding or not st.session_state.llm:
                embedding, llm = initialize_azure_models()
                if not embedding or not llm:
                    return
                st.session_state.embedding = embedding
                st.session_state.llm = llm
            
            vectorstore = load_vector_store(db_name, st.session_state.embedding)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.current_database = db_name
                
                metadata = load_database_metadata(db_name)
                st.session_state.processed_files = list(metadata.keys())
                st.session_state.file_metadata = metadata
                
                file_contents = load_saved_files(db_name)
                st.session_state.file_contents = file_contents
                
                if file_contents:
                    st.success(f"✅ Database '{db_name}' loaded successfully with {len(file_contents)} files!")
                else:
                    st.warning("⚠️ Database loaded but no file contents found. PDF viewing will not work.")
                
                st.rerun()
            else:
                st.error(f"❌ Failed to load database '{db_name}'")
                
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")

def process_new_database(db_name, uploaded_files, export_type):
    """Process and create a new database with parallel processing"""
    try:
        # Setup logging
        logger, log_file = setup_logging(db_name)
        logger.info(f"🚀 Starting database creation: {db_name}")
        logger.info(f"📁 Processing {len(uploaded_files)} files using {NUM_WORKERS} parallel workers")
        
        with st.spinner("Processing documents and creating database..."):
            # Initialize models
            embedding, llm = initialize_azure_models()
            if not embedding or not llm:
                logger.error("Failed to initialize Azure OpenAI models")
                return
            
            # Save files and prepare for parallel processing
            file_paths_info = []
            file_contents = {}
            files_info = {}
            
            for uploaded_file in uploaded_files:
                temp_path, content = save_file_permanently(uploaded_file, db_name)
                if temp_path and content:
                    file_paths_info.append((temp_path, uploaded_file.name))
                    file_contents[uploaded_file.name] = content
                    files_info[uploaded_file.name] = {
                        'size': len(content),
                        'type': uploaded_file.type,
                        'path': temp_path
                    }
                    logger.info(f"📄 Saved file: {uploaded_file.name} ({len(content)} bytes)")
            
            if file_paths_info:
                # Process documents in parallel
                st.info(f"🔄 Processing {len(file_paths_info)} documents using {NUM_WORKERS} parallel workers...")
                splits, processing_stats = process_documents_parallel(file_paths_info, export_type, logger)
                
                if splits:
                    # Create vector store
                    st.info("🔗 Creating vector database...")
                    vectorstore = create_and_save_vector_store(splits, embedding, db_name, logger)
                    
                    if vectorstore:
                        # Update session state
                        st.session_state.vectorstore = vectorstore
                        st.session_state.current_database = db_name
                        st.session_state.processed_files = list(file_contents.keys())
                        st.session_state.file_contents = file_contents
                        st.session_state.file_metadata = files_info
                        st.session_state.embedding = embedding
                        st.session_state.llm = llm
                        
                        # Save metadata with processing stats
                        save_database_metadata(db_name, files_info, processing_stats)
                        
                        logger.info(f"🎉 Database '{db_name}' created successfully!")
                        logger.info(f"📊 Total chunks generated: {len(splits)}")
                        logger.info(f"📋 Log file saved: {log_file}")
                        
                        st.success(f"✅ Database '{db_name}' created successfully!")
                        st.info(f"📊 Processed {len(uploaded_files)} documents into {len(splits)} chunks using parallel processing")
                        
                        # Display processing summary
                        with st.expander("📈 Processing Summary", expanded=True):
                            successful = sum(1 for stats in processing_stats.values() if stats['success'])
                            total_time = sum(stats['processing_time'] for stats in processing_stats.values())
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Files Processed", f"{successful}/{len(uploaded_files)}")
                            with col2:
                                st.metric("Total Chunks", len(splits))
                            with col3:
                                st.metric("Processing Time", f"{total_time:.1f}s")
                            
                            # Show per-file stats
                            st.markdown("**Per-file Processing Times:**")
                            for filename, stats in processing_stats.items():
                                status = "✅" if stats['success'] else "❌"
                                st.write(f"{status} {filename}: {stats['processing_time']:.2f}s ({stats['chunks_count']} chunks)")
                        
                        st.rerun()
                    else:
                        logger.error("Failed to create vector store")
                        st.error("❌ Failed to create vector store")
                else:
                    logger.error("No document splits generated")
                    st.error("❌ No document splits were generated")
            else:
                logger.error("No valid files to process")
                st.error("❌ No valid files to process")
                
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
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
    st.markdown(f"*Advanced document processing with parallel processing ({NUM_WORKERS} workers) and comprehensive logging*")
    
    # Check if PDF viewer is active
    if st.session_state.pdf_viewer_active and st.session_state.current_pdf_file and st.session_state.current_pdf_page:
        display_pdf_page(st.session_state.current_pdf_file, st.session_state.current_pdf_page)
    else:
        # Only show chat interface if we have a database loaded
        if st.session_state.vectorstore and st.session_state.current_database:
            display_chat_history()
            
            # Input section
            st.subheader("🔍 Ask a Question")
            
            with st.form(key='question_form'):
                question = st.text_input(
                    "Your question:",
                    placeholder="What would you like to know about your documents?",
                    help="Ask any question about your uploaded documents"
                )
                search_button = st.form_submit_button("🚀 Ask", type="primary", use_container_width=False)

            # Execute search
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
                        
                        response = rag_chain.invoke({"input": question})
                        
                        add_to_chat_history(
                            question=question,
                            answer=response["answer"],
                            context_docs=response.get("context", [])
                        )
                        
                        st.session_state.last_processed_question = ""
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    
        else:
            # Welcome message
            st.info("👋 Welcome to the Document Intelligence Platform with Parallel Processing. Please upload documents or load an existing database from the sidebar to get started.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### ⚡ Parallel Processing")
                st.write(f"Process documents using {NUM_WORKERS} parallel workers for faster document ingestion and OCR.")
            with col2:
                st.markdown("### 📊 Comprehensive Logging")
                st.write("Detailed logging for each file processing time, success/failure status, and performance metrics.")
            with col3:
                st.markdown("### 🎯 Intelligent Search")
                st.write("Natural language queries with precise source attribution and PDF page viewing.")
            
            # Show system information
            with st.expander("🔧 System Information", expanded=False):
                st.write(f"**CPU Cores Available:** {mp.cpu_count()}")
                st.write(f"**Parallel Workers:** {NUM_WORKERS}")
                st.write(f"**Database Directory:** {DATABASE_DIR}")
                st.write(f"**Files Directory:** {FILES_DIR}")
                st.write(f"**Logs Directory:** {LOGS_DIR}")

if __name__ == "__main__":
    main()