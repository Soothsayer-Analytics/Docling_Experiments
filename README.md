## Docling Experiments

This repository contains experimental code for document intelligence using Docling, Azure OpenAI, and Streamlit. The projects explore different approaches to document processing, chunking, OCR, and question-answering.

## Project Structure

### Core Scripts

1. **Final.py** - Main command-line document processing pipeline
   - Uses Docling for document parsing and OCR
   - Integrates with Azure OpenAI for embeddings and GPT-4
   - Creates vector stores using Milvus
   - Supports both chunk-based and markdown-based processing
   - Includes rate limiting and enhanced OCR support

2. **test.py** - Streamlit web application (basic version)
   - Interactive document upload and processing
   - Real-time Q&A interface
   - PDF page viewing capability
   - Simple document management

3. **test_v1.py** - Command-line version with markdown processing
   - Similar to Final.py but with different configuration
   - Focuses on markdown export type
   - Includes comprehensive logging

4. **test_v2.py** - Streamlit app with parallel processing (multiprocessing)
   - Uses ProcessPoolExecutor for parallel document processing
   - Enhanced metadata standardization
   - Comprehensive processing logs
   - Performance metrics display

5. **test_v3.py** - Streamlit app with thread-based parallel processing
   - Uses ThreadPoolExecutor for better compatibility
   - Fallback to sequential processing if parallel fails
   - Event loop management for async operations
   - Thread usage breakdown

### Supporting Scripts

6. **app.py** - Additional Streamlit application (older version)
7. **app_v3.py** - Another variant of Streamlit app
8. **document_chunk.py** - Document chunking utilities
9. **main.py** - Main entry point (possibly older pipeline)
10. **rough.py** - Experimental code snippets
11. **stacked_classifier.py** - Classification experiments

### Configuration

- **.env** - Environment variables for Azure OpenAI configuration (secrets redacted)

## Key Features

- **Document Processing**: Support for PDF, DOCX, TXT, and MD files
- **OCR Integration**: Automatic OCR for scanned PDFs using Docling
- **Parallel Processing**: Multi-threaded/multi-process document ingestion
- **Vector Storage**: Milvus-based vector stores for efficient similarity search
- **Interactive Q&A**: Natural language querying with source attribution
- **PDF Viewer**: Integrated PDF page viewing with highlighted references
- **Comprehensive Logging**: Detailed processing logs with performance metrics

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Bhanuprakash9391/Docling_Experiments.git
   cd Docling_Experiments
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Azure OpenAI:
   - Update `.env` with your Azure OpenAI endpoint and API key
   - Ensure you have appropriate Azure resources provisioned

4. Run the desired application:
   - Command-line: `python Final.py`
   - Streamlit: `streamlit run test.py`

## Dependencies

- Python 3.8+
- Docling & LangChain
- Azure OpenAI SDK
- Streamlit (for web applications)
- Milvus (vector database)
- PyMuPDF (PDF processing)

## Note

All secrets (API keys, connection strings) have been redacted and replaced with placeholders. Update the configuration files with your own credentials before running the applications.

## License

This project is for experimental purposes. Use at your own discretion.
