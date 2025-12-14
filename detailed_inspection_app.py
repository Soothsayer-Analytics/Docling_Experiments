import streamlit as st
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.base_models import InputFormat

# Page Config
st.set_page_config(
    page_title="PDF Image & Table Counter",
    page_icon="�",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def count_images_and_tables(file_path):
    """
    Analyze PDF and count images and tables using multiple verification methods
    Returns: dict with verified counts
    """
    # Configure Docling pipeline for accurate detection
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
        mode="accurate"  # Use accurate mode for better detection
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    start_time = time.time()
    result = converter.convert(file_path)
    processing_time = time.time() - start_time
    
    # Method 1: Count from document structure (most reliable)
    image_count_method1 = 0
    table_count_method1 = 0
    
    try:
        # Access the document's items directly
        if hasattr(result.document, 'items'):
            for item in result.document.items:
                item_type = str(type(item).__name__).lower()
                if 'picture' in item_type or 'image' in item_type:
                    image_count_method1 += 1
                elif 'table' in item_type:
                    table_count_method1 += 1
    except Exception as e:
        st.warning(f"Method 1 (document structure) failed: {e}")
    
    # Method 2: Count from markdown output
    md_text = result.document.export_to_markdown()
    
    # Count image markers in markdown
    image_count_method2 = md_text.count('<!-- image -->')
    
    # Count tables in markdown (look for table separator pattern)
    table_count_method2 = 0
    lines = md_text.split('\n')
    for line in lines:
        stripped = line.strip()
        # Table header separator pattern (e.g., |---|---|)
        if '|' in stripped and '--' in stripped:
            # Verify it's a table separator (has at least 2 columns)
            if stripped.count('|') >= 3:  # At least |--|--| 
                table_count_method2 += 1
    
    # Method 3: Count from JSON export (additional verification)
    image_count_method3 = 0
    table_count_method3 = 0
    
    try:
        json_output = result.document.export_to_dict()
        
        # Recursively search for images and tables in JSON structure
        def count_in_dict(obj, img_count, tbl_count):
            if isinstance(obj, dict):
                # Check for type indicators
                obj_type = obj.get('type', '').lower()
                if 'picture' in obj_type or 'image' in obj_type:
                    img_count += 1
                elif 'table' in obj_type:
                    tbl_count += 1
                
                # Recurse through all values
                for value in obj.values():
                    img_count, tbl_count = count_in_dict(value, img_count, tbl_count)
            elif isinstance(obj, list):
                for item in obj:
                    img_count, tbl_count = count_in_dict(item, img_count, tbl_count)
            
            return img_count, tbl_count
        
        image_count_method3, table_count_method3 = count_in_dict(json_output, 0, 0)
    except Exception as e:
        st.warning(f"Method 3 (JSON structure) failed: {e}")
    
    # Double-check: Use the maximum count from all methods for accuracy
    # This ensures we don't miss any detections
    final_image_count = max(image_count_method1, image_count_method2, image_count_method3)
    final_table_count = max(table_count_method1, table_count_method2, table_count_method3)
    
    # Store all counts for verification display
    verification_data = {
        'images': {
            'method1_structure': image_count_method1,
            'method2_markdown': image_count_method2,
            'method3_json': image_count_method3,
            'final': final_image_count
        },
        'tables': {
            'method1_structure': table_count_method1,
            'method2_markdown': table_count_method2,
            'method3_json': table_count_method3,
            'final': final_table_count
        },
        'processing_time': processing_time
    }
    
    return verification_data

def main():
    st.title("🔢 PDF Image & Table Counter")
    st.markdown("""
    Upload a PDF to get accurate counts of **images** and **tables**.
    
    Uses **triple verification** to ensure accuracy:
    - ✅ Document structure analysis
    - ✅ Markdown output parsing
    - ✅ JSON structure verification
    """)
    
    uploaded_file = st.file_uploader("📄 Upload PDF", type=['pdf'])
    
    if not uploaded_file:
        st.info("� Upload a PDF file to start counting")
        return
    
    st.divider()
    
    # Process the document
    file_path = save_uploaded_file(uploaded_file)
    
    try:
        with st.spinner("🔍 Analyzing PDF with triple verification..."):
            results = count_images_and_tables(file_path)
        
        st.success(f"✅ Analysis complete in {results['processing_time']:.2f} seconds")
        
        st.divider()
        
        # Display final counts prominently
        st.subheader("📊 Detection Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="�️ Images Detected",
                value=results['images']['final'],
                help="Total number of images found in the PDF"
            )
        
        with col2:
            st.metric(
                label="📋 Tables Detected",
                value=results['tables']['final'],
                help="Total number of tables found in the PDF"
            )
        
        st.divider()
        
        # Show verification details
        with st.expander("🔍 View Verification Details", expanded=False):
            st.markdown("### Triple Verification Breakdown")
            st.markdown("*The final count uses the maximum from all methods to ensure accuracy*")
            
            st.markdown("#### 🖼️ Image Detection Methods:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Method 1",
                    results['images']['method1_structure'],
                    help="Document structure analysis"
                )
            with col2:
                st.metric(
                    "Method 2",
                    results['images']['method2_markdown'],
                    help="Markdown output parsing"
                )
            with col3:
                st.metric(
                    "Method 3",
                    results['images']['method3_json'],
                    help="JSON structure verification"
                )
            
            # Check if all methods agree
            img_counts = [
                results['images']['method1_structure'],
                results['images']['method2_markdown'],
                results['images']['method3_json']
            ]
            if len(set(img_counts)) == 1:
                st.success("✅ All methods agree on image count")
            else:
                st.warning(f"⚠️ Methods differ - using maximum count ({results['images']['final']}) for accuracy")
            
            st.divider()
            
            st.markdown("#### 📋 Table Detection Methods:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Method 1",
                    results['tables']['method1_structure'],
                    help="Document structure analysis"
                )
            with col2:
                st.metric(
                    "Method 2",
                    results['tables']['method2_markdown'],
                    help="Markdown output parsing"
                )
            with col3:
                st.metric(
                    "Method 3",
                    results['tables']['method3_json'],
                    help="JSON structure verification"
                )
            
            # Check if all methods agree
            tbl_counts = [
                results['tables']['method1_structure'],
                results['tables']['method2_markdown'],
                results['tables']['method3_json']
            ]
            if len(set(tbl_counts)) == 1:
                st.success("✅ All methods agree on table count")
            else:
                st.warning(f"⚠️ Methods differ - using maximum count ({results['tables']['final']}) for accuracy")
    
    except Exception as e:
        import traceback
        st.error(f"❌ An error occurred: {str(e)}")
        with st.expander("View error details"):
            st.code(traceback.format_exc())
    
    finally:
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass

if __name__ == "__main__":
    main()
