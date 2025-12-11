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
PDF_FOLDER_PATH = "/home/azureuser/Bhanu/Docling/BAS"  # Change this to your PDF folder path
DATABASE_NAME = "FullData"     # Change this to your desired database name
LOAD_DATABASE_NAME = None          # Set this to load existing database instead of creating new one
SINGLE_QUERY = None                # Set this to execute a single query
USE_INTERACTIVE_MODE = True        # Set to True to start interactive mode after processing
EXPORT_TYPE = "markdown"             # Choose: "chunks" or "markdown"

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
        """Setup logging configuration"""
        log_filename = LOGS_DIR / f"document_processing_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create custom logger
        logger = logging.getLogger('document_processor')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_filename)
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

    def process_documents(self, file_paths, export_type, db_name):
        """Process documents using Docling"""
        try:
            self.logger.info(f"Starting document processing for database '{db_name}' with {len(file_paths)} files")
            self.logger.info(f"Export type: {export_type}")
            
            # Initialize Docling loader
            if export_type == ExportType.DOC_CHUNKS:
                self.logger.info("Using HybridChunker for document processing")
                chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")
                loader = DoclingLoader(
                    file_path=file_paths,
                    export_type=export_type,
                    chunker=chunker,
                )
            else:
                self.logger.info("Using standard Docling loader")
                loader = DoclingLoader(
                    file_path=file_paths,
                    export_type=export_type,
                )
            
            # Load documents with timing
            docs_start_time = time.time()
            docs = loader.load()
            docs_load_time = time.time() - docs_start_time
            
            self.logger.info(f"Docling loaded {len(docs)} document chunks in {docs_load_time:.2f} seconds")
            
            # Log statistics about loaded documents
            total_content_length = sum(len(doc.page_content) for doc in docs)
            avg_content_length = total_content_length / len(docs) if docs else 0
            
            self.logger.info(f"Document statistics: Total content length: {total_content_length:,} characters, "
                           f"Average chunk length: {avg_content_length:.0f} characters")
            
            # Process based on export type
            if export_type == ExportType.DOC_CHUNKS:
                splits = docs
                self.logger.info("Using DOC_CHUNKS - no additional splitting required")
            elif export_type == ExportType.MARKDOWN:
                self.logger.info("Processing MARKDOWN export - applying header-based splitting")
                splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "Header_1"),
                        ("##", "Header_2"),
                        ("###", "Header_3"),
                    ],
                )
                splits_start_time = time.time()
                splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
                splits_time = time.time() - splits_start_time
                self.logger.info(f"Markdown splitting completed in {splits_time:.2f} seconds, created {len(splits)} chunks")
            else:
                splits = docs
                self.logger.info("Using default processing - no additional splitting")
            
            # Log file-specific information
            file_sources = {}
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                filename = Path(source).name if source != 'Unknown' else 'Unknown'
                if filename not in file_sources:
                    file_sources[filename] = {'chunks': 0, 'total_chars': 0}
                file_sources[filename]['chunks'] += 1
                file_sources[filename]['total_chars'] += len(doc.page_content)
            
            for filename, stats in file_sources.items():
                self.logger.info(f"File '{filename}': {stats['chunks']} chunks, {stats['total_chars']:,} characters")
            
            self.logger.info(f"Document processing completed: {len(splits)} total chunks created")
            return splits
            
        except Exception as e:
            self.logger.error(f"Error processing documents for database '{db_name}': {str(e)}")
            print(f"Error processing documents: {str(e)}")
            return None

    def create_and_save_vector_store(self, splits, embedding, db_name):
        """Create vector store and save it persistently"""
        try:
            self.logger.info(f"Creating vector store for database '{db_name}' with {len(splits)} chunks")
            
            # Setup event loop for async operations
            self.setup_event_loop()
            
            db_path = DATABASE_DIR / f"{db_name}.db"
            
            # Use synchronous approach to avoid async issues
            vectorstore_start_time = time.time()
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
            vectorstore_time = time.time() - vectorstore_start_time
            
            self.logger.info(f"Vector store created successfully in {vectorstore_time:.2f} seconds")
            self.logger.info(f"Database saved at: {db_path}")
            
            return vectorstore
        except Exception as e:
            self.logger.error(f"Error creating vector store for '{db_name}': {str(e)}")
            print(f"Error creating vector store: {str(e)}")
            return None

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

    def create_database_from_folder(self, folder_path, db_name, export_type=ExportType.DOC_CHUNKS):
        """Create a new database from a folder of PDFs"""
        try:
            print(f"\n📁 Processing folder: {folder_path}")
            print(f"🗄️ Creating database: {db_name}")
            
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
            
            # Process documents
            print("⚙️ Processing documents with Docling...")
            splits = self.process_documents(file_paths, export_type, db_name)
            
            if not splits:
                return False
            
            # Create vector store
            print("🔍 Creating vector store...")
            vectorstore = self.create_and_save_vector_store(splits, embedding, db_name)
            
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
    """Main function with hardcoded configuration"""
    processor = DocumentProcessor()
    
    # Show available databases
    print("📑 PDF Document Processing System")
    print("="*50)
    
    databases = processor.get_available_databases()
    if databases:
        print("📊 Available databases:")
        for i, db in enumerate(databases, 1):
            print(f"  {i}. {db}")
        print()
    
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
        success = processor.create_database_from_folder(PDF_FOLDER_PATH, DATABASE_NAME, export_type)
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
