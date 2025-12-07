import os
import json
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
import time
import logging
from datetime import datetime
from typing import List
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

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

# Configuration - Hardcoded paths and settings
PDF_FOLDER_PATH = "/home/azureuser/Bhanu/Docling/Data"  # Change this to your PDF folder path
DATABASE_NAME = "Final"     # Change this to your desired database name
LOAD_DATABASE_NAME = None          # Set this to load existing database instead of creating new one
SINGLE_QUERY = None                # Set this to execute a single query
USE_INTERACTIVE_MODE = True        # Set to True to start interactive mode after processing
EXPORT_TYPE = "chunks"             # Choose: "chunks" or "markdown"

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

class DocumentProcessor:
    def __init__(self):
        self.vectorstore = None
        self.llm = None
        self.embedding = None
        self.current_database = None
        self.processed_files = []
        self.file_contents = {}
        self.file_metadata = {}
        self.top_k = TOP_K
        self.logger = self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration with immediate flushing"""
        log_filename = LOGS_DIR / f"document_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create custom logger
        logger = logging.getLogger('document_processor')
        logger.setLevel(logging.INFO)
        
        # Create file handler with immediate flushing
        file_handler = logging.FileHandler(log_filename, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Force immediate flushing
        logger.propagate = False
        
        return logger

    def log_processing_time(self, func):
        """Decorator to log processing time for functions"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                self.logger.info(f"Function '{func.__name__}' completed in {duration:.2f} seconds")
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                self.logger.error(f"Function '{func.__name__}' failed after {duration:.2f} seconds: {str(e)}")
                raise
        return wrapper

    def setup_event_loop(self):
        """Setup event loop for async operations"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def get_available_databases(self):
        """Get list of available databases"""
        databases = []
        for db_path in DATABASE_DIR.glob("*.db"):
            db_name = db_path.stem
            databases.append(db_name)
        return databases

    def save_database_metadata(self, db_name, files_info):
        """Save metadata about the database"""
        metadata_path = DATABASE_DIR / f"{db_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(files_info, f, indent=2)

    def load_database_metadata(self, db_name):
        """Load metadata about the database"""
        metadata_path = DATABASE_DIR / f"{db_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def initialize_azure_models(self):
        """Initialize Azure OpenAI models"""
        try:
            self.logger.info("Initializing Azure OpenAI models...")
            
            embedding = AzureOpenAIEmbeddings(
                azure_deployment=AZURE_CONFIG["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
                openai_api_version=AZURE_CONFIG["OPENAI_API_VERSION"],
            )
            
            llm = AzureChatOpenAI(
                azure_deployment=AZURE_CONFIG["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                openai_api_version=AZURE_CONFIG["OPENAI_API_VERSION"],
                temperature=0,
            )
            
            self.logger.info("Azure OpenAI models initialized successfully")
            return embedding, llm
        except Exception as e:
            self.logger.error(f"Error initializing Azure OpenAI models: {str(e)}")
            print(f"Error initializing Azure OpenAI models: {str(e)}")
            return None, None

    def get_pdf_files_from_folder(self, folder_path):
        """Get all PDF files from a folder (both .pdf and .PDF extensions)"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            self.logger.error(f"Folder does not exist: {folder_path}")
            print(f"Error: Folder '{folder_path}' does not exist")
            return []
        
        # Get both .pdf and .PDF files
        pdf_files_lower = list(folder_path.glob("*.pdf"))
        pdf_files_upper = list(folder_path.glob("*.PDF"))
        pdf_files = pdf_files_lower + pdf_files_upper
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in folder: {folder_path}")
            print(f"Warning: No PDF files found in folder '{folder_path}'")
        else:
            self.logger.info(f"Found {len(pdf_files)} PDF files in folder: {folder_path}")
            print(f"Found {len(pdf_files)} PDF files in folder: {folder_path}")
        
        return pdf_files

    def copy_files_to_database(self, pdf_files, db_name):
        """Copy PDF files to database directory"""
        try:
            self.logger.info(f"Copying {len(pdf_files)} files for database '{db_name}'")
            
            db_files_dir = FILES_DIR / db_name
            db_files_dir.mkdir(exist_ok=True)
            
            file_paths = []
            file_contents = {}
            files_info = {}
            
            for pdf_file in pdf_files:
                # Copy file to database directory
                dest_path = db_files_dir / pdf_file.name
                shutil.copy2(pdf_file, dest_path)
                
                # Read file content
                with open(pdf_file, 'rb') as f:
                    content = f.read()
                
                file_paths.append(str(dest_path))
                file_contents[pdf_file.name] = content
                files_info[pdf_file.name] = {
                    'size': len(content),
                    'type': 'application/pdf',
                    'original_path': str(pdf_file)
                }
                
                self.logger.info(f"Copied file '{pdf_file.name}' (Size: {len(content):,} bytes)")
            
            self.logger.info(f"Successfully copied {len(pdf_files)} files")
            return file_paths, file_contents, files_info
        
        except Exception as e:
            self.logger.error(f"Error copying files: {str(e)}")
            print(f"Error copying files: {str(e)}")
            return [], {}, {}

    def diagnose_pdf_with_docling_ocr(self, file_paths):
        """Enhanced PDF diagnosis for Docling OCR capabilities"""
        for file_path in file_paths:
            filename = Path(file_path).name
            file_size = Path(file_path).stat().st_size
            
            self.logger.info(f"Docling OCR Diagnosis for {filename} (Size: {file_size:,} bytes)")
            
            if file_size == 0:
                self.logger.warning(f"File {filename} is empty on disk")
                continue
            
            try:
                # Check PDF structure with PyMuPDF first
                doc = fitz.open(file_path)
                page_count = len(doc)
                
                if page_count == 0:
                    self.logger.warning(f"PDF {filename} has 0 pages")
                    doc.close()
                    continue
                
                # Enhanced content analysis
                has_text = False
                has_images = False
                total_text_chars = 0
                total_images = 0
                
                # Check more pages for better analysis
                pages_to_check = min(5, page_count)  # Check first 5 pages
                for page_num in range(pages_to_check):
                    page = doc[page_num]
                    text = page.get_text().strip()
                    images = page.get_images()
                    
                    if text:
                        has_text = True
                        total_text_chars += len(text)
                    if images:
                        has_images = True
                        total_images += len(images)
                
                doc.close()
                
                avg_text_per_page = total_text_chars / pages_to_check if pages_to_check > 0 else 0
                avg_images_per_page = total_images / pages_to_check if pages_to_check > 0 else 0
                
                # Enhanced PDF classification
                if has_text and avg_text_per_page > 500 and not has_images:
                    self.logger.info(f"✅ {filename}: Text-rich PDF - Fast processing expected ({avg_text_per_page:.0f} chars/page)")
                elif has_text and avg_text_per_page > 200 and has_images:
                    self.logger.info(f"📄 {filename}: Mixed content PDF - Moderate OCR processing ({avg_text_per_page:.0f} chars/page, {avg_images_per_page:.1f} images/page)")
                elif has_text and avg_text_per_page <= 200 and has_images:
                    self.logger.info(f"🖼️ {filename}: Image-heavy PDF - Intensive OCR processing expected ({avg_text_per_page:.0f} chars/page, {avg_images_per_page:.1f} images/page)")
                elif not has_text and has_images:
                    self.logger.info(f"📸 {filename}: Fully scanned PDF - Maximum OCR processing time ({avg_images_per_page:.1f} images/page)")
                else:
                    self.logger.warning(f"❓ {filename}: Unclear PDF structure - processing may vary")
                
                # Additional file size warning
                if file_size > 50 * 1024 * 1024:  # 50MB
                    self.logger.warning(f"⚠️ {filename}: Large file ({file_size / (1024*1024):.1f}MB) - OCR processing will take significant time")
                    
            except Exception as e:
                self.logger.error(f"Failed to diagnose {filename}: {str(e)}")

    def process_with_enhanced_ocr_support(self, file_paths, export_type, db_name, max_chunk_size=1000):
        """Enhanced processing with OCR support and no timeout restrictions"""
        try:
            self.logger.info(f"Starting enhanced OCR processing for database '{db_name}' with {len(file_paths)} files")
            self.logger.info("No timeout restrictions - allowing full OCR processing time")
            
            all_splits = []
            failed_files = []
            empty_files = []
            ocr_files = []
            
            def process_single_file(file_path, file_index):
                filename = Path(file_path).name
                
                try:
                    self.logger.info(f"Processing file {file_index}/{len(file_paths)}: {filename}")
                    file_start_time = time.time()
                    
                    # Configure Docling with OCR support
                    if export_type == ExportType.DOC_CHUNKS:
                        chunker = HybridChunker(
                            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
                            max_chunk_size=max_chunk_size,
                            chunk_overlap=100
                        )
                        loader = DoclingLoader(
                            file_path=[file_path],
                            export_type=export_type,
                            chunker=chunker,
                        )
                    else:
                        loader = DoclingLoader(
                            file_path=[file_path],
                            export_type=export_type,
                        )
                    
                    # Load document (this includes OCR for scanned PDFs) - NO TIMEOUT
                    self.logger.info(f"Starting Docling OCR processing for {filename}...")
                    docs = loader.load()
                    
                    file_time = time.time() - file_start_time
                    self.logger.info(f"Docling processing completed for {filename} in {file_time:.2f} seconds")
                    
                    return filename, docs, None, file_time
                    
                except Exception as e:
                    file_time = time.time() - file_start_time if 'file_start_time' in locals() else 0
                    error_msg = str(e)
                    self.logger.error(f"Error processing {filename} after {file_time:.2f} seconds: {error_msg}")
                    return filename, None, error_msg, file_time
            
            # Process files sequentially without timeout
            for i, file_path in enumerate(file_paths, 1):
                filename = Path(file_path).name
                
                try:
                    # Process file without timeout restrictions
                    filename, docs, error, processing_time = process_single_file(file_path, i)
                    
                    if error:
                        self.logger.error(f"Error processing {filename}: {error}")
                        failed_files.append(filename)
                        continue
                    
                    if not docs or all(not doc.page_content.strip() for doc in docs):
                        self.logger.warning(f"No content extracted from {filename}")
                        empty_files.append(filename)
                        continue
                    
                    # Process the documents
                    if export_type == ExportType.DOC_CHUNKS:
                        file_splits = docs
                    elif export_type == ExportType.MARKDOWN:
                        splitter = MarkdownHeaderTextSplitter(
                            headers_to_split_on=[
                                ("#", "Header_1"),
                                ("##", "Header_2"),
                                ("###", "Header_3"),
                            ],
                            strip_headers=False
                        )
                        
                        file_splits = []
                        for doc in docs:
                            if doc.page_content.strip():
                                doc_splits = splitter.split_text(doc.page_content)
                                
                                for split in doc_splits:
                                    if len(split.page_content) > max_chunk_size:
                                        content = split.page_content
                                        for j in range(0, len(content), max_chunk_size):
                                            chunk_content = content[j:j + max_chunk_size]
                                            if chunk_content.strip():
                                                chunk_doc = Document(
                                                    page_content=chunk_content,
                                                    metadata={**split.metadata, 'chunk_index': j // max_chunk_size}
                                                )
                                                file_splits.append(chunk_doc)
                                    else:
                                        if split.page_content.strip():
                                            file_splits.append(split)
                    else:
                        file_splits = docs
                    
                    total_chars = sum(len(doc.page_content) for doc in file_splits)
                    
                    # Determine if OCR was likely used based on processing time and content characteristics
                    avg_char_per_page = total_chars / len(docs) if docs else 0
                    if processing_time > 30:  # Files taking longer likely used OCR
                        ocr_files.append(filename)
                        self.logger.info(f"OCR File {filename}: {len(file_splits)} chunks, {total_chars:,} characters, processed in {processing_time:.2f} seconds")
                    else:
                        self.logger.info(f"Text File {filename}: {len(file_splits)} chunks, {total_chars:,} characters, processed in {processing_time:.2f} seconds")
                    
                    all_splits.extend(file_splits)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {filename}: {str(e)}")
                    failed_files.append(filename)
                
                # Force log flush after each file
                for handler in self.logger.handlers:
                    handler.flush()
            
            # Final summary
            total_content_length = sum(len(doc.page_content) for doc in all_splits)
            avg_content_length = total_content_length / len(all_splits) if all_splits else 0
            
            self.logger.info(f"Enhanced OCR Processing Summary:")
            self.logger.info(f"   Successfully processed: {len(file_paths) - len(failed_files) - len(empty_files)} files")
            self.logger.info(f"   OCR processed files: {len(ocr_files)} files")
            self.logger.info(f"   Empty/no content: {len(empty_files)} files")
            self.logger.info(f"   Failed: {len(failed_files)} files")
            self.logger.info(f"   Total chunks: {len(all_splits)}")
            self.logger.info(f"   Total content: {total_content_length:,} characters")
            self.logger.info(f"   Average chunk size: {avg_content_length:.0f} characters")
            
            if ocr_files:
                self.logger.info(f"OCR processed files: {', '.join(ocr_files)}")
            if empty_files:
                self.logger.warning(f"Empty files: {', '.join(empty_files)}")
            if failed_files:
                self.logger.error(f"Failed files: {', '.join(failed_files)}")
            
            return all_splits
            
        except Exception as e:
            self.logger.error(f"Error in enhanced OCR processing for database '{db_name}': {str(e)}")
            print(f"Error processing documents: {str(e)}")
            return None

    def create_and_save_vector_store_with_rate_limiting(self, splits, embedding, db_name, batch_size=50, delay_between_batches=60):
        """Create and save vector store with rate limiting for API quotas"""
        try:
            self.logger.info(f"Creating vector store with rate limiting for database '{db_name}'")
            self.logger.info(f"Processing {len(splits)} chunks in batches of {batch_size} with {delay_between_batches}s delays")
            
            # Setup event loop for async operations
            self.setup_event_loop()
            
            db_path = DATABASE_DIR / f"{db_name}.db"
            
            # Initialize Milvus vector store
            vectorstore = Milvus(
                embedding_function=embedding,
                collection_name=f"collection_{db_name}",
                connection_args={
                    "uri": str(db_path),
                    "token": ""  # Empty token to force sync mode
                },
            )
            
            # Process documents in batches to respect rate limits
            total_batches = (len(splits) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(splits))
                batch = splits[start_idx:end_idx]
                
                batch_start_time = time.time()
                
                try:
                    self.logger.info(f"Processing batch {batch_num + 1}/{total_batches}: chunks {start_idx + 1}-{end_idx}")
                    
                    # Add documents to vector store
                    vectorstore.add_documents(batch)
                    
                    batch_time = time.time() - batch_start_time
                    self.logger.info(f"Batch {batch_num + 1} completed in {batch_time:.2f} seconds")
                    
                    # Add delay between batches (except for the last batch)
                    if batch_num < total_batches - 1:
                        self.logger.info(f"Waiting {delay_between_batches} seconds before next batch...")
                        time.sleep(delay_between_batches)
                    
                except Exception as e:
                    batch_time = time.time() - batch_start_time if 'batch_start_time' in locals() else 0
                    self.logger.error(f"Error processing batch {batch_num + 1} after {batch_time:.2f} seconds: {str(e)}")
                    
                    # For rate limit errors, wait longer and retry once
                    if "rate" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                        self.logger.warning(f"Rate limit detected, waiting {delay_between_batches * 2} seconds before retry...")
                        time.sleep(delay_between_batches * 2)
                        
                        try:
                            self.logger.info(f"Retrying batch {batch_num + 1}/{total_batches}")
                            vectorstore.add_documents(batch)
                            retry_time = time.time() - batch_start_time
                            self.logger.info(f"Batch {batch_num + 1} retry completed in {retry_time:.2f} seconds")
                        except Exception as retry_error:
                            self.logger.error(f"Retry failed for batch {batch_num + 1}: {str(retry_error)}")
                            raise retry_error
                    else:
                        raise e
            
            self.logger.info(f"Vector store creation completed for database '{db_name}'")
            self.logger.info(f"Successfully processed {len(splits)} chunks in {total_batches} batches")
            
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error creating vector store with rate limiting for '{db_name}': {str(e)}")
            print(f"Error creating vector store: {str(e)}")
            return None

    def create_database_from_folder_with_rate_limiting(self, folder_path, db_name, export_type=ExportType.DOC_CHUNKS, batch_size=50, delay_between_batches=60):
        """Create a new database from a folder of PDFs with rate limiting and unlimited OCR processing time"""
        try:
            print(f"\n📁 Processing folder: {folder_path}")
            print(f"🗄️ Creating database: {db_name}")
            print(f"⏱️ Using batch processing: {batch_size} chunks per batch, {delay_between_batches}s delay")
            print("🔓 No timeout restrictions - allowing full OCR processing time")
            
            # Get PDF files from folder
            pdf_files = self.get_pdf_files_from_folder(folder_path)
            if not pdf_files:
                return False
            
            # Initialize Azure models
            print("🔧 Initializing Azure OpenAI models...")
            embedding, llm = self.initialize_azure_models()
            if not embedding or not llm:
                return False
            
            self.embedding = embedding
            self.llm = llm
            
            # Copy files to database directory
            print("📋 Copying files to database directory...")
            file_paths, file_contents, files_info = self.copy_files_to_database(pdf_files, db_name)
            
            if not file_paths:
                return False
            
            # Diagnose potential issues with PDFs and OCR capabilities
            print("🔍 Diagnosing PDF files for Docling OCR...")
            self.diagnose_pdf_with_docling_ocr(file_paths)
            
            # Process documents with enhanced OCR support (no timeout)
            print("⚙️ Processing documents with Docling OCR (unlimited time)...")
            splits = self.process_with_enhanced_ocr_support(file_paths, export_type, db_name, max_chunk_size=1000)
            
            if not splits:
                return False
            
            # Create vector store with rate limiting
            print(f"🔍 Creating vector store with rate limiting ({len(splits)} chunks)...")
            vectorstore = self.create_and_save_vector_store_with_rate_limiting(splits, embedding, db_name, batch_size, delay_between_batches)
            
            if not vectorstore:
                return False
            
            # Update instance variables
            self.vectorstore = vectorstore
            self.current_database = db_name
            self.processed_files = list(file_contents.keys())
            self.file_contents = file_contents
            self.file_metadata = files_info
            
            # Save metadata
            self.save_database_metadata(db_name, files_info)
            
            print(f"✅ Database '{db_name}' created successfully!")
            print(f"📊 Processed {len(pdf_files)} PDF files into {len(splits)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating database from folder: {str(e)}")
            print(f"❌ Error creating database: {str(e)}")
            return False

    def load_vector_store(self, db_name, embedding):
        """Load existing vector store"""
        try:
            self.logger.info(f"Loading vector store for database '{db_name}'")
            
            # Setup event loop for async operations
            self.setup_event_loop()
            
            db_path = DATABASE_DIR / f"{db_name}.db"
            if not db_path.exists():
                self.logger.warning(f"Database file not found: {db_path}")
                return None
            
            vectorstore = Milvus(
                embedding_function=embedding,
                collection_name=f"collection_{db_name}",
                connection_args={
                    "uri": str(db_path),
                    "token": ""  # Add empty token to force sync mode
                },
            )
            
            self.logger.info(f"Vector store loaded successfully for database '{db_name}'")
            return vectorstore
        except Exception as e:
            self.logger.error(f"Error loading vector store for '{db_name}': {str(e)}")
            print(f"Error loading vector store: {str(e)}")
            return None

    def create_database_from_folder_with_rate_limiting(self, folder_path, db_name, export_type=ExportType.DOC_CHUNKS, batch_size=50, delay_between_batches=60):
        """Create a new database from a folder of PDFs with rate limiting and OCR support"""
        try:
            print(f"\n📁 Processing folder: {folder_path}")
            print(f"🗄️ Creating database: {db_name}")
            print(f"⏱️ Using batch processing: {batch_size} chunks per batch, {delay_between_batches}s delay")
            
            # Get PDF files from folder
            pdf_files = self.get_pdf_files_from_folder(folder_path)
            if not pdf_files:
                return False
            
            # Initialize Azure models
            print("🔧 Initializing Azure OpenAI models...")
            embedding, llm = self.initialize_azure_models()
            if not embedding or not llm:
                return False
            
            self.embedding = embedding
            self.llm = llm
            
            # Copy files to database directory
            print("📋 Copying files to database directory...")
            file_paths, file_contents, files_info = self.copy_files_to_database(pdf_files, db_name)
            
            if not file_paths:
                return False
            
            # Diagnose potential issues with PDFs and OCR capabilities
            print("🔍 Diagnosing PDF files for Docling OCR...")
            self.diagnose_pdf_with_docling_ocr(file_paths)
            
            # Process documents with enhanced OCR support
            print("⚙️ Processing documents with Docling OCR (enhanced)...")
            splits = self.process_with_enhanced_ocr_support(file_paths, export_type, db_name, max_chunk_size=1000)
            
            if not splits:
                return False
            
            # Create vector store with rate limiting
            print(f"🔍 Creating vector store with rate limiting ({len(splits)} chunks)...")
            vectorstore = self.create_and_save_vector_store_with_rate_limiting(splits, embedding, db_name, batch_size, delay_between_batches)
            
            if not vectorstore:
                return False
            
            # Update instance variables
            self.vectorstore = vectorstore
            self.current_database = db_name
            self.processed_files = list(file_contents.keys())
            self.file_contents = file_contents
            self.file_metadata = files_info
            
            # Save metadata
            self.save_database_metadata(db_name, files_info)
            
            print(f"✅ Database '{db_name}' created successfully!")
            print(f"📊 Processed {len(pdf_files)} PDF files into {len(splits)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating database from folder: {str(e)}")
            print(f"❌ Error creating database: {str(e)}")
            return False

    def load_existing_database(self, db_name):
        """Load an existing database"""
        try:
            print(f"📂 Loading database: {db_name}")
            
            # Initialize models if not already done
            if not self.embedding or not self.llm:
                print("🔧 Initializing Azure OpenAI models...")
                embedding, llm = self.initialize_azure_models()
                if not embedding or not llm:
                    return False
                self.embedding = embedding
                self.llm = llm
            
            # Load vector store
            vectorstore = self.load_vector_store(db_name, self.embedding)
            if not vectorstore:
                return False
            
            self.vectorstore = vectorstore
            self.current_database = db_name
            
            # Load metadata
            metadata = self.load_database_metadata(db_name)
            self.processed_files = list(metadata.keys())
            self.file_metadata = metadata
            
            print(f"✅ Database '{db_name}' loaded successfully!")
            print(f"📄 Contains {len(self.processed_files)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading database '{db_name}': {str(e)}")
            print(f"❌ Error loading database: {str(e)}")
            return False

    def query_database(self, question, top_k=None):
        """Query the loaded database"""
        if not self.vectorstore:
            print("❌ No database loaded. Please load or create a database first.")
            return None
        
        if not question.strip():
            print("❌ Please provide a question to search for.")
            return None
        
        try:
            if top_k is None:
                top_k = self.top_k
            
            self.logger.info(f"Executing query: '{question[:100]}...' for database '{self.current_database}'")
            print(f"\n🔍 Searching for: {question}")
            
            query_start_time = time.time()
            
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": top_k}
            )
            question_answer_chain = create_stuff_documents_chain(
                self.llm, PROMPT
            )
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # Invoke the chain
            response = rag_chain.invoke({"input": question})
            
            query_time = time.time() - query_start_time
            self.logger.info(f"Query completed in {query_time:.2f} seconds")
            
            # Display results
            print(f"\n💡 Answer:")
            print("=" * 50)
            print(response["answer"])
            print("=" * 50)
            
            # Display source references
            if "context" in response and response["context"]:
                print(f"\n📚 Source References:")
                print("-" * 30)
                
                for i, doc in enumerate(response["context"], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    filename = Path(source).name if source != 'Unknown' else 'Unknown'
                    page = doc.metadata.get('page', doc.metadata.get('page_number', 1))
                    
                    print(f"\n{i}. 📄 {filename} (Page {page})")
                    preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    print(f"   Preview: {preview}")
                    print("-" * 30)
            
            return response
            
        except Exception as e:
            query_time = time.time() - query_start_time if 'query_start_time' in locals() else 0
            self.logger.error(f"Query failed after {query_time:.2f} seconds: {str(e)}")
            print(f"❌ Error processing query: {str(e)}")
            return None

    def interactive_mode(self):
        """Interactive mode for querying"""
        if not self.vectorstore:
            print("❌ No database loaded. Please load or create a database first.")
            return
        
        print(f"\n🤖 Interactive Query Mode")
        print(f"📊 Database: {self.current_database}")
        print(f"📄 Documents: {len(self.processed_files)}")
        print("Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            try:
                question = input("🔍 Enter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                self.query_database(question)
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main function with rate limiting configuration"""
    processor = DocumentProcessor()
    
    # Show available databases
    print("📑 PDF Document Processing System (with Enhanced OCR Support)")
    print("="*70)
    
    databases = processor.get_available_databases()
    if databases:
        print("📊 Available databases:")
        for i, db in enumerate(databases, 1):
            print(f"  {i}. {db}")
        print()
    
    # Configuration for rate limiting
    BATCH_SIZE = 25  # Reduced batch size for S0 tier
    DELAY_BETWEEN_BATCHES = 30  # Slightly longer delay
    
    # Determine export type
    export_type = ExportType.DOC_CHUNKS if EXPORT_TYPE == 'chunks' else ExportType.MARKDOWN
    
    # Load existing database if specified
    if LOAD_DATABASE_NAME:
        success = processor.load_existing_database(LOAD_DATABASE_NAME)
        if not success:
            print("Failed to load database. Exiting.")
            return
    
    # Create new database from folder if specified
    elif PDF_FOLDER_PATH and DATABASE_NAME:
        success = processor.create_database_from_folder_with_rate_limiting(
            PDF_FOLDER_PATH, 
            DATABASE_NAME, 
            export_type, 
            batch_size=BATCH_SIZE, 
            delay_between_batches=DELAY_BETWEEN_BATCHES
        )
        if not success:
            print("Failed to create database. Exiting.")
            return
    
    else:
        print("❌ Error: Please configure either PDF_FOLDER_PATH + DATABASE_NAME or LOAD_DATABASE_NAME")
        print("\nConfiguration options at the top of the script:")
        print("  PDF_FOLDER_PATH - Path to folder containing PDF files")
        print("  DATABASE_NAME - Name for new database")
        print("  LOAD_DATABASE_NAME - Name of existing database to load")
        return
    
    # Execute single query if specified
    if SINGLE_QUERY:
        processor.query_database(SINGLE_QUERY)
    
    # Start interactive mode if enabled
    if USE_INTERACTIVE_MODE:
        processor.interactive_mode()

if __name__ == "__main__":
    main()