# Detailed Project Portfolio

This document provides comprehensive details about all major projects, including repository information, technology stack, key features, and project structure.

---

## 1. Docling Experiments

### Repository Information
- **Repository Name**: Docling_Experiments
- **GitHub Link**: https://github.com/Bhanuprakash9391/Docling_Experiments
- **Creation Date**: December 7, 2025
- **Status**: Active Development
- **Visibility**: Public

### Project Overview
A comprehensive suite of document intelligence experiments leveraging Docling's OCR capabilities combined with Azure OpenAI for advanced document processing and question-answering.

### Technology Stack
- **Programming Language**: Python 3.8+
- **Document Processing**: Docling, PyMuPDF (fitz), PIL
- **AI/ML**: Azure OpenAI (GPT-4, text-embedding-ada-002), LangChain
- **Vector Database**: Milvus
- **Web Framework**: Streamlit
- **Parallel Processing**: ThreadPoolExecutor, ProcessPoolExecutor
- **Logging**: Python logging with file handlers

### Key Files and Their Purpose

#### Core Processing Pipelines
1. **Final.py** - Main command-line document processing pipeline
   - Batch processing of PDF folders with OCR support
   - Rate-limited API calls for Azure OpenAI
   - Enhanced OCR diagnostics and processing
   - Vector store creation with Milvus
   - Interactive query mode

2. **test.py** - Basic Streamlit web application
   - Document upload and processing interface
   - Real-time Q&A with source attribution
   - PDF page viewer with image extraction
   - Chat history management

3. **test_v1.py** - Command-line version with markdown processing
   - Focus on markdown export type
   - Comprehensive logging and timing metrics
   - File-specific processing statistics

4. **test_v2.py** - Streamlit app with multiprocessing
   - Parallel document processing using ProcessPoolExecutor
   - Metadata standardization for Milvus compatibility
   - Processing logs with performance metrics
   - Thread-safe operations

5. **test_v3.py** - Streamlit app with thread-based parallelism
   - ThreadPoolExecutor for better Streamlit compatibility
   - Fallback to sequential processing
   - Event loop management for async operations
   - Thread usage breakdown

#### Supporting Scripts
- **app.py** / **app_v3.py** - Alternative Streamlit implementations
- **document_chunk.py** - Document chunking utilities
- **main.py** - Legacy main entry point
- **rough.py** - Experimental code snippets
- **stacked_classifier.py** - Classification experiments

### Key Features
- **Automatic OCR**: Docling handles scanned PDFs without manual configuration
- **Parallel Processing**: Up to 4 workers for simultaneous document ingestion
- **Rate Limiting**: Configurable batch sizes and delays for API quota management
- **Comprehensive Logging**: Detailed processing logs with timing metrics
- **PDF Viewer**: Integrated page viewing with highlighted references
- **Metadata Standardization**: Ensures consistent schema for vector storage
- **Secrets Management**: All API keys redacted and replaced with placeholders

### Directory Structure
```
Docling_Experiments/
├── .env                    # Environment variables (secrets redacted)
├── .gitignore             # Git ignore rules for data/logs
├── README.md              # Project documentation
├── Final.py               # Main CLI pipeline
├── test.py                # Basic Streamlit app
├── test_v1.py             # CLI with markdown
├── test_v2.py             # Streamlit with multiprocessing
├── test_v3.py             # Streamlit with threading
├── app.py                 # Alternative Streamlit app
├── app_v3.py              # Another variant
├── document_chunk.py      # Chunking utilities
├── main.py                # Legacy entry point
├── rough.py               # Experimental code
└── stacked_classifier.py  # Classification experiments
```

### Usage Scenarios
1. **Document Q&A System**: Upload technical manuals and ask specific questions
2. **Research Paper Analysis**: Process academic papers and extract key insights
3. **Compliance Document Review**: Search through regulatory documents
4. **Technical Manual Navigation**: Quickly find information in complex manuals

---

## 2. Dexko Backup Codes

### Repository Information
- **Repository Name**: Dexko_Backup_Codes
- **GitHub Link**: https://github.com/Bhanuprakash9391/Dexko_Backup_Codes
- **Status**: Archive/Backup
- **Visibility**: Public

### Project Overview
A consolidated backup repository containing multiple Dexko-related projects spanning AI implementations, support experiments, and industrial equipment code.

### Subprojects

#### DexKoAI_Code
- **Purpose**: AI and machine learning implementations for Dexko manufacturing applications
- **Key Areas**:
  - Predictive maintenance algorithms
  - Quality control using computer vision
  - Production optimization through reinforcement learning
  - Anomaly detection in sensor data
- **Technologies**: Python, TensorFlow/PyTorch, Scikit-learn, OpenCV

#### DexKo_Support_Experiments
- **Purpose**: Experimental code for customer support automation
- **Key Areas**:
  - Chatbot development for technical support
  - Knowledge base management systems
  - Ticket classification and routing
  - Customer sentiment analysis
- **Technologies**: Natural Language Processing, Dialogflow, FastAPI

#### DexKoAI_IE_07_01_25
- **Purpose**: Industrial equipment AI implementations (July 1, 2025)
- **Key Areas**:
  - Equipment monitoring and predictive maintenance
  - Failure prediction models
  - Maintenance scheduling optimization
  - Energy consumption analysis
- **Technologies**: Time series analysis, IoT data processing, ML forecasting

### Repository Structure
```
Dexko_Backup_Codes/
├── DexKoAI_Code/
│   ├── README.md
│   ├── models/              # Trained ML models
│   ├── notebooks/           # Jupyter notebooks
│   └── src/                 # Source code
├── DexKo_Support_Experiments/
│   ├── README.md
│   ├── chatbots/            # Chatbot implementations
│   ├── knowledge_base/      # Knowledge management
│   └── analytics/           # Support analytics
└── DexKoAI_IE_07_01_25/
    ├── README.md
    ├── equipment_monitoring/
    ├── predictive_maintenance/
    └── maintenance_scheduling/
```

---

## 3. IDP India Projects

### Repository Information
- **Repository Name**: IDP_India_Projects
- **GitHub Link**: https://github.com/Bhanuprakash9391/IDP_India_Projects
- **Status**: Active
- **Visibility**: Public

### Project Overview
Intelligent Document Processing solutions tailored for the Indian market, addressing unique challenges like multilingual documents, handwritten text, and specific industry requirements.

### Key Features
- **Multilingual Support**: Hindi, English, and regional language processing
- **Handwritten Text Recognition**: Specialized models for Indian handwriting
- **Industry-Specific Templates**: Banking, healthcare, government forms
- **Compliance**: GDPR, Indian data protection regulations
- **Cloud Integration**: Azure, AWS, and GCP deployment options

### Technology Stack
- **OCR**: Tesseract, Google Vision, Custom models
- **NLP**: spaCy, Transformers for Indian languages
- **Backend**: FastAPI, Django
- **Database**: PostgreSQL, MongoDB
- **Frontend**: React, Streamlit

---

## 4. Bahrain Air Services Demo

### Repository Information
- **Repository Name**: Baharain_Air_Services_Demo
- **GitHub Link**: https://github.com/Bhanuprakash9391/Baharain_Air_Services_Demo
- **Status**: Demonstration
- **Visibility**: Public

### Project Overview
A demonstration project for Bahrain Air Services showcasing document processing capabilities for flight operations, customer service, and compliance.

### Key Components
1. **Flight Document Processing**: Automated processing of flight manifests, schedules
2. **Customer Service Automation**: Chatbot for flight information and booking
3. **Compliance Document Management**: Regulatory document processing and archiving
4. **Analytics Dashboard**: Flight performance and customer satisfaction metrics

### Technologies
- **Frontend**: Streamlit dashboard
- **Backend**: Python FastAPI
- **Database**: SQLite for demo, scalable to PostgreSQL
- **AI**: Document classification, entity extraction

---

## 5. I2POC Contract Versions

### Repository Information
- **Repository Name**: I2POC_Contract_Versions
- **GitHub Link**: https://github.com/Bhanuprakash9391/I2POC_Contract_Versions
- **Status**: Version Control
- **Visibility**: Public

### Project Overview
Version control repository for Idea-to-POC contract documents, legal agreements, and compliance materials.

### Contents
- **Contract Templates**: Standard agreements for different project types
- **Legal Documents**: Compliance, NDAs, IP agreements
- **Version History**: Track changes across contract versions
- **Approval Workflows**: Document review and approval processes

### Technology
- **Version Control**: Git for document tracking
- **Document Comparison**: diff tools for contract changes
- **Workflow Management**: Basic approval system

---

## 6. Idea to POC

### Repository Information
- **Repository Name**: Idea_to_POC
- **GitHub Link**: https://github.com/Bhanuprakash9391/Idea_to_POC
- **Status**: Active Development
- **Visibility**: Public

### Project Overview
A full-stack application for managing the complete idea-to-proof-of-concept lifecycle, from idea submission through evaluation, development, and final delivery.

### Key Modules
1. **Idea Submission Portal**: Users submit ideas with details and attachments
2. **Evaluation Engine**: AI-powered idea scoring and feasibility analysis
3. **Development Tracking**: Progress monitoring, resource allocation
4. **Resource Estimation**: AI agents for time and cost estimation
5. **Reviewer Dashboard**: Panel for idea review and approval
6. **Catalog System**: Browse and search existing ideas and POCs

### Technology Stack
- **Frontend**: Streamlit for rapid prototyping
- **Backend**: Python with FastAPI
- **Database**: SQLite (development), PostgreSQL (production)
- **AI Integration**: Azure OpenAI for scoring and estimation
- **Authentication**: JWT-based auth system

### Key Files (from I2POC_Streamlit project)
- **app.py**: Main Streamlit application
- **models.py**: Database models and schemas
- **services/**: AI scoring, research agents, workflow orchestration
- **pages/**: Idea submission, catalog, development, reviewer dashboard
- **utils/**: Helpers, cache management, error handling

---

## 7. AgenticAI

### Repository Information
- **Repository Name**: AgenticAI
- **GitHub Link**: https://github.com/Bhanuprakash9391/AgenticAI
- **Status**: Framework Development
- **Visibility**: Public

### Project Overview
An agentic AI framework for autonomous task execution, enabling multi-agent systems that can collaborate to solve complex problems.

### Core Concepts
- **Multi-Agent Systems**: Specialized agents for different tasks
- **Workflow Orchestration**: Coordinating agent interactions
- **Task Decomposition**: Breaking complex problems into manageable tasks
- **Knowledge Sharing**: Agents learning from each other's experiences

### Applications
- **Automated Research**: Agents gathering and synthesizing information
- **Code Generation**: AI pair programming assistants
- **Process Automation**: End-to-end business process automation
- **Decision Support**: Multi-perspective analysis for complex decisions

---

## 8. J&J Codes Backup

### Repository Information
- **Repository Name**: J-J_Codes_BackUp
- **GitHub Link**: https://github.com/Bhanuprakash9391/J-J_Codes_BackUp
- **Status**: Backup/Archive
- **Visibility**: Private

### Project Overview
Backup repository for Johnson & Johnson related codebases, containing healthcare analytics, pharmaceutical data processing, and compliance systems.

### Key Areas
1. **Healthcare Analytics**: Patient data analysis, treatment effectiveness
2. **Pharmaceutical Data**: Drug interaction analysis, clinical trial data
3. **Compliance Systems**: Regulatory compliance automation
4. **Quality Control**: Manufacturing quality assurance algorithms

### Security Notes
- Repository is private due to sensitive healthcare data
- All sensitive information has been redacted
- Used for backup and disaster recovery purposes

---

## Development Practices Across Projects

### Code Quality
- **Documentation**: Comprehensive README files for each project
- **Modular Design**: Separation of concerns, reusable components
- **Error Handling**: Robust exception handling and logging
- **Testing**: Unit tests for critical functionality (where applicable)

### Security
- **Secrets Management**: Environment variables, never hardcoded
- **API Key Rotation**: Regular rotation of sensitive credentials
- **Access Control**: Appropriate repository visibility settings
- **Data Protection**: Redaction of sensitive information before commits

### Deployment
- **Containerization**: Docker support for key projects
- **Cloud Ready**: Designed for deployment on Azure/AWS/GCP
- **Scalability**: Horizontal scaling considerations
- **Monitoring**: Built-in logging and performance metrics

---

## Contact and Collaboration

For more information about any project:
- **GitHub Issues**: Open issues on respective repositories
- **Email**: bhanuprakash9391@gmail.com
- **Documentation**: Refer to individual project README files

## Future Roadmap
1. **Docling Experiments**: Add support for more document types, improve parallel processing
2. **Idea to POC**: Enhance AI scoring algorithms, add collaboration features
3. **IDP India Projects**: Expand language support, add industry-specific modules
4. **AgenticAI**: Develop more specialized agents, improve orchestration

---
*Last Updated: December 7, 2025*
*Document Version: 2.0*
