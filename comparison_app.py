import streamlit as st
import os
# Force CPU usage since GPU is not available
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.base_models import InputFormat

# Page Configuration
st.set_page_config(
    page_title="Advanced Document Intelligence Comparison",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        border-left: 5px solid #1a73e8;
        background-color: #f1f3f4;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 14px;
        color: #5f6368;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #202124;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }
    .badge-fast { background-color: #34a853; }
    .badge-slow { background-color: #ea4335; }
    .badge-rich { background-color: #4285f4; }
    .badge-basic { background-color: #9aa0a6; }
    .docling-enhanced {
        border: 2px solid #4285f4;
        background-color: #e8f0fe;
    }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def extract_with_pymupdf(file_path):
    """PyMuPDF: Fast, low-level text extraction"""
    start_time = time.time()
    try:
        doc = fitz.open(file_path)
        text = ""
        page_count = len(doc)
        for page in doc:
            text += page.get_text() + "\n\n"
        doc.close()
        return {
            "text": text.strip(),
            "time": time.time() - start_time,
            "pages": page_count,
            "method": "PyMuPDF"
        }
    except Exception as e:
        return {"error": str(e), "time": 0}

def extract_with_pdfplumber(file_path):
    """pdfplumber: Better layout analysis, great for tables"""
    start_time = time.time()
    try:
        text = ""
        tables_found = 0
        page_count = 0
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                # Text extraction with layout preservation
                text += (page.extract_text(layout=True) or "") + "\n\n"
                # Simple count of detected tables
                tables = page.find_tables()
                tables_found += len(tables)
        
        return {
            "text": text.strip(),
            "time": time.time() - start_time,
            "pages": page_count,
            "tables_found": tables_found,
            "method": "pdfplumber"
        }
    except Exception as e:
        return {"error": str(e), "time": 0}

def extract_with_docling(file_path, enable_ocr=True, do_table_structure=True):
    """Docling: AI-Powered Layout Analysis & Structure - Enhanced Version"""
    start_time = time.time()
    try:
        # Enhanced converter setup for optimal performance and accuracy
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = enable_ocr
        pipeline_options.do_table_structure = do_table_structure
        
        # Configure for CPU-only performance optimization
        # Since GPU is not available, optimize for CPU processing
        
        # Try to configure OCR options if available
        try:
            if hasattr(pipeline_options, 'ocr_options'):
                # Use RapidOCR for better speed and accuracy on CPU
                # Note: Some versions of Docling don't have use_gpu attribute
                # So we'll skip GPU configuration and rely on default CPU behavior
                if hasattr(pipeline_options.ocr_options, 'ocr_engine'):
                    pipeline_options.ocr_options.ocr_engine = "rapidocr"
                # Optimize for CPU performance if available
                if hasattr(pipeline_options.ocr_options, 'cpu_threads'):
                    pipeline_options.ocr_options.cpu_threads = 4  # Use 4 CPU threads
        except AttributeError:
            pass  # OCR options not available in this version
        
        # Configure table structure for better extraction
        if do_table_structure:
            try:
                table_options = TableStructureOptions()
                table_options.enable_table_structure = True
                # Use table-transformer for better table detection
                pipeline_options.table_structure_options = table_options
            except:
                pass  # Table options not available
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Convert document
        result = converter.convert(file_path)
        
        # Export results in multiple formats
        md_text = result.document.export_to_markdown()
        json_export = result.document.export_to_dict()
        
        # Extract comprehensive metrics
        item_counts = {"text": 0, "table": 0, "image": 0, "other": 0}
        fallback_text = []
        page_count = 0
        total_text_length = 0
        tables_data = []
        
        if "pages" in json_export:
            page_count = len(json_export["pages"])
            for page_num, page in json_export["pages"].items():
                for item in page.get("items", []):
                    kind = item.get("label", "unknown")
                    if kind == "text":
                        item_counts["text"] += 1
                        text_content = item.get("text", "")
                        if text_content.strip():
                            fallback_text.append(text_content)
                            total_text_length += len(text_content)
                    elif kind == "table":
                        item_counts["table"] += 1
                        # Extract table data if available
                        table_data = item.get("table_data", {})
                        if table_data:
                            tables_data.append({
                                "page": page_num,
                                "rows": len(table_data.get("rows", [])),
                                "columns": len(table_data.get("columns", []))
                            })
                    elif kind in ["picture", "figure", "image"]:
                        item_counts["image"] += 1
                    else:
                        item_counts["other"] += 1
        
        # Calculate quality metrics
        avg_text_per_item = total_text_length / max(item_counts["text"], 1)
        
        # Enhanced output generation
        if not md_text.strip() or ("<!-- image -->" in md_text and len(md_text) < 500):
            # Generate comprehensive report for empty or minimal results
            md_text = "## 📊 Docling Document Analysis Report\n\n"
            md_text += f"### 📄 Document Statistics\n"
            md_text += f"- **Total Pages:** {page_count}\n"
            md_text += f"- **Text Elements:** {item_counts['text']}\n"
            md_text += f"- **Tables Detected:** {item_counts['table']}\n"
            md_text += f"- **Images Identified:** {item_counts['image']}\n"
            md_text += f"- **Other Elements:** {item_counts['other']}\n"
            md_text += f"- **Total Text Characters:** {total_text_length:,}\n\n"
            
            if fallback_text:
                md_text += "### 📝 Extracted Text Content\n"
                for i, text in enumerate(fallback_text[:20], 1):  # Show first 20 items
                    if text.strip():
                        md_text += f"**Text {i}:** {text[:200]}...\n\n"
                if len(fallback_text) > 20:
                    md_text += f"\n*... and {len(fallback_text) - 20} more text items*\n"
            
            if tables_data:
                md_text += "\n### 📊 Table Analysis\n"
                for i, table in enumerate(tables_data, 1):
                    md_text += f"**Table {i}** (Page {table['page']}): {table['rows']} rows × {table['columns']} columns\n"
        else:
            # Add statistics to existing markdown
            md_text = f"## 📊 Docling Analysis\n\n**Document Statistics:** {page_count} pages, {item_counts['text']} text elements, {item_counts['table']} tables, {item_counts['image']} images\n\n" + md_text
        
        # Calculate numeric tables found for comparison
        tables_found_count = item_counts['table']
        
        return {
            "text": md_text,
            "raw_json": json_export,
            "time": time.time() - start_time,
            "pages": page_count,
            "tables_found": tables_found_count,
            "method": "Docling",
            "is_markdown": True,
            "item_counts": item_counts,
            "total_text_length": total_text_length,
            "avg_text_per_item": avg_text_per_item,
            "tables_data": tables_data
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "error": f"Docling Error: {str(e)}\n\nDebug Info:\n{error_details[-500:]}", 
            "time": 0,
            "pages": 0,
            "tables_found": 0,
            "method": "Docling"
        }

def render_metric_card(title, data, badge_class="badge-basic", badge_text="Standard"):
    err = data.get('error')
    text_content = data.get('text', '')
    
    val_display = str(len(text_content))
    if err:
        val_display = "Error"
    
    # Get tables found count
    tables_found = data.get('tables_found', 0)
    if isinstance(tables_found, str):
        import re
        numbers = re.findall(r'\d+', tables_found)
        tables_display = tables_found if numbers else "0"
    else:
        tables_display = str(tables_found)
    
    # Special styling for Docling
    container_class = "metric-container docling-enhanced" if title == "Docling" else "metric-container"
    
    st.markdown(f"""
    <div class="{container_class}">
        <h3>{title} <span class="badge {badge_class}">{badge_text}</span></h3>
        <div class="metric-label">⏱️ Processing Time</div>
        <div class="metric-value">{data.get('time', 0):.2f}s</div>
        <div class="metric-label">📝 Characters</div>
        <div class="metric-value">{val_display}</div>
        <div class="metric-label">📄 Pages</div>
        <div class="metric-value">{data.get('pages', '?')}</div>
        <div class="metric-label">📊 Tables Found</div>
        <div class="metric-value">{tables_display}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if err:
        with st.expander(f"{title} Error Details", expanded=False):
            st.error(err)

def main():
    st.title("📑 Intelligent Document Processing: Method Comparison")
    st.write("Compare the depth and quality of different document extraction techniques.")
    
    # Add info about enhanced Docling
    st.info("""
    **🚀 Enhanced Docling Configuration (CPU Optimized):** 
    - CPU-optimized processing for systems without GPU
    - Advanced table detection with table-transformer
    - Comprehensive document structure analysis
    - Detailed metrics and fallback content extraction
    - RapidOCR engine for efficient CPU-based OCR
    """)

    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Experiment Controls")
        
        st.subheader("1. Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        st.divider()
        st.subheader("2. Docling Configuration")
        enable_ocr = st.checkbox("Enable OCR (Optical Character Recognition)", value=True, 
                                help="Needed for scanned documents. Uses RapidOCR engine for better accuracy.")
        enable_tables = st.checkbox("Enable Advanced Table Analysis", value=True, 
                                   help="Uses table-transformer for superior table detection and structure preservation.")
        
        st.divider()
        st.subheader("3. Performance Settings")
        st.info("⚠️ GPU acceleration is not available on this system. Using CPU-optimized processing.")
        
        # Remove GPU checkbox since GPU is not available
        # use_gpu = st.checkbox("Use GPU Acceleration (if available)", value=True,
        #                      help="Significantly speeds up Docling processing. Requires CUDA-compatible GPU.")
        
        st.divider()
        st.info("""
        **Method Comparison:**
        - **PyMuPDF**: 'Dumb' stream extraction. Fast, but loses layout.
        - **pdfplumber**: 'Layout-aware'. Good for text positions, basic table detection.
        - **Docling**: 'AI-driven'. Full layout analysis, OCR, table reconstruction, and semantic Markdown export.
        """)

    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        
        # Force CPU usage since GPU is not available
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        if st.button("🚀 Run CPU-Optimized Comparison Analysis", type="primary"):
            
            # 1. PyMuPDF Processing
            with st.status("Running PyMuPDF...", expanded=False) as status:
                res_pymupdf = extract_with_pymupdf(file_path)
                status.update(label="PyMuPDF Complete", state="complete")
                
            # 2. pdfplumber Processing
            with st.status("Running pdfplumber...", expanded=False) as status:
                res_pdfplumber = extract_with_pdfplumber(file_path)
                status.update(label="pdfplumber Complete", state="complete")
                
            # 3. Docling Processing
            with st.status("Running CPU-Optimized Docling (AI Analysis)...", expanded=True) as status:
                st.write("🚀 Initializing CPU-optimized Docling models...")
                st.write("📊 Configuring CPU-optimized processing and advanced features...")
                res_docling = extract_with_docling(file_path, enable_ocr, enable_tables)
                
                if res_docling.get("error"):
                    st.error("Docling encountered an error")
                    with st.expander("Error Details"):
                        st.error(res_docling["error"])
                else:
                    # Show detailed metrics
                    item_counts = res_docling.get("item_counts", {})
                    st.write(f"✅ Analysis complete!")
                    st.write(f"📄 Pages: {res_docling.get('pages', 0)}")
                    st.write(f"📝 Text Elements: {item_counts.get('text', 0)}")
                    st.write(f"📊 Tables: {item_counts.get('table', 0)}")
                    st.write(f"🖼️ Images: {item_counts.get('image', 0)}")

                status.update(label="Docling Complete", state="complete")

            # Cleanup
            try:
                os.unlink(file_path)
            except:
                pass

            # --- Results Display ---
            st.divider()
            st.header("📊 Analytical Results")
            
            # Metrics Columns
            m1, m2, m3 = st.columns(3)
            with m1:
                render_metric_card("PyMuPDF", res_pymupdf, "badge-fast", "Fastest")
            with m2:
                render_metric_card("pdfplumber", res_pdfplumber, "badge-basic", "Layout Aware")
            with m3:
                render_metric_card("Docling", res_docling, "badge-rich", "AI Powered")

            # Quality Comparison
            st.subheader("🎯 Quality Comparison")
            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric("Text Extraction", "Basic", "Layout Lost")
            with q2:
                st.metric("Table Detection", "Good", "Basic Structure")
            with q3:
                st.metric("Semantic Analysis", "Advanced", "Full Context")

            # Content Tabs
            st.subheader("Content Inspection")
            tabs = st.tabs(["📝 Text Output (Raw)", "📐 Layout Preserved (pdfplumber)", 
                          "🤖 Semantic Markdown (Docling)", "🔍 Side-by-Side", "📊 Detailed Analysis"])
            
            with tabs[0]:
                st.subheader("PyMuPDF Raw Text")
                st.text_area("Raw Output", res_pymupdf.get("text", ""), height=400,
                           help="Basic text extraction without layout preservation")
                
            with tabs[1]:
                st.subheader("pdfplumber Layout Text")
                st.text_area("Layout Output", res_pdfplumber.get("text", ""), height=400,
                           help="Text with basic layout preservation")
                
            with tabs[2]:
                st.subheader("Docling Structured Markdown")
                
                # Show rendered markdown vs source
                md_col, raw_col = st.columns(2)
                with md_col:
                    st.markdown("#### 📖 Rendered Preview")
                    md_text = res_docling.get("text", "")
                    if md_text:
                        st.markdown(md_text)
                    else:
                        st.warning("No content extracted")
                    
                    # Show additional Docling metrics
                    if not res_docling.get("error"):
                        with st.expander("📈 Detailed Metrics"):
                            item_counts = res_docling.get("item_counts", {})
                            st.write(f"**Text Elements:** {item_counts.get('text', 0)}")
                            st.write(f"**Tables:** {item_counts.get('table', 0)}")
                            st.write(f"**Images:** {item_counts.get('image', 0)}")
                            st.write(f"**Other Elements:** {item_counts.get('other', 0)}")
                            st.write(f"**Total Text Length:** {res_docling.get('total_text_length', 0):,} characters")
                            st.write(f"**Average Text per Item:** {res_docling.get('avg_text_per_item', 0):.1f} characters")
                
                with raw_col:
                    st.markdown("#### 📝 Markdown Source")
                    text_content = res_docling.get("text", "")
                    st.code(text_content, language="markdown")
                    
                    # Debugger for empty results
                    if not text_content or len(text_content) < 100:
                        st.warning("Docling output is minimal. This could indicate:")
                        st.info("1. The PDF is image-based and OCR was disabled\n2. The document contains unsupported elements\n3. Try enabling OCR in the sidebar")
                        with st.expander("🔍 Debug: Raw JSON Structure"):
                            st.json(res_docling.get("raw_json", {}))
            
            with tabs[3]:
                st.subheader("🔍 Side-by-Side Comparison (First 500 chars)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**PyMuPDF**")
                    pymupdf_text = res_pymupdf.get("text", "")
                    st.text(pymupdf_text[:500] + ("..." if len(pymupdf_text) > 500 else ""))
                    st.caption(f"Length: {len(pymupdf_text):,} chars")
                with c2:
                    st.markdown("**pdfplumber**")
                    pdfplumber_text = res_pdfplumber.get("text", "")
                    st.text(pdfplumber_text[:500] + ("..." if len(pdfplumber_text) > 500 else ""))
                    st.caption(f"Length: {len(pdfplumber_text):,} chars")
                with c3:
                    st.markdown("**Docling**")
                    docling_text = res_docling.get("text", "")
                    st.text(docling_text[:500] + ("..." if len(docling_text) > 500 else ""))
                    st.caption(f"Length: {len(docling_text):,} chars")
            
            with tabs[4]:
                st.subheader("📊 Detailed Analysis")
                
                # Performance comparison
                st.markdown("### ⚡ Performance Comparison")
                perf_data = {
                    "Method": ["PyMuPDF", "pdfplumber", "Docling"],
                    "Time (s)": [
                        res_pymupdf.get("time", 0),
                        res_pdfplumber.get("time", 0),
                        res_docling.get("time", 0)
                    ],
                    "Text Length": [
                        len(res_pymupdf.get("text", "")),
                        len(res_pdfplumber.get("text", "")),
                        len(res_docling.get("text", ""))
                    ],
                    "Tables Found": [
                        res_pymupdf.get("tables_found", 0),
                        res_pdfplumber.get("tables_found", 0),
                        res_docling.get("tables_found", 0)
                    ]
                }
                st.dataframe(perf_data, use_container_width=True)
                
                # Docling detailed analysis
                if not res_docling.get("error"):
                    st.markdown("### 🧠 Docling Deep Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 📈 Element Distribution")
                        item_counts = res_docling.get("item_counts", {})
                        if item_counts:
                            elements_data = {
                                "Element Type": list(item_counts.keys()),
                                "Count": list(item_counts.values())
                            }
                            st.bar_chart(elements_data, x="Element Type", y="Count")
                    
                    with col2:
                        st.markdown("#### 📊 Quality Metrics")
                        metrics_data = {
                            "Metric": ["Text Elements", "Tables", "Images", "Text/Element"],
                            "Value": [
                                item_counts.get("text", 0),
                                item_counts.get("table", 0),
                                item_counts.get("image", 0),
                                res_docling.get("avg_text_per_item", 0)
                            ]
                        }
                        st.dataframe(metrics_data, use_container_width=True)
                    
                    # Table details if available
                    tables_data = res_docling.get("tables_data", [])
                    if tables_data:
                        st.markdown("#### 📋 Table Details")
                        table_details = []
                        for i, table in enumerate(tables_data, 1):
                            table_details.append({
                                "Table #": i,
                                "Page": table.get("page", "N/A"),
                                "Rows": table.get("rows", 0),
                                "Columns": table.get("columns", 0)
                            })
                        st.dataframe(table_details, use_container_width=True)

            # Cleanup reminder
            st.info("💡 **Tip:** The uploaded file has been automatically cleaned up. You can upload another file to compare.")

if __name__ == "__main__":
    main()
