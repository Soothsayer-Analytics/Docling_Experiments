import streamlit as st
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.base_models import InputFormat
import fitz  # PyMuPDF
from PIL import Image
import io

# Page Config
st.set_page_config(
    page_title="PDF Image & Table Counter",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def check_if_scanned(file_path):
    """
    Check if a PDF is likely scanned (high image area, low text).
    Returns: (is_scanned, details_string)
    """
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        scanned_pages = 0
        
        for page in doc:
            # excessive logic: low text length and presence of images covering page
            text = page.get_text()
            images = page.get_images()
            
            # Simple heuristic: If text is very sparse (< 50 chars) and there are images
            # or if the page is just one big image
            if len(text.strip()) < 50:
                if len(images) > 0:
                    scanned_pages += 1
                else:
                    # No text, no images? might be a drawing or unrecognized content.
                    # But for "scanned" usually means images.
                    pass
        
        doc.close()
        
        if total_pages > 0 and (scanned_pages / total_pages) > 0.5:
             return True, f"{scanned_pages}/{total_pages} pages appear to be scanned images."
        return False, "Document appears to contain digital text."
        
    except Exception as e:
        return False, f"Could not determine scanned status: {e}"

def count_images_and_tables(file_path):
    """
    Analyze PDF and count images and tables using multiple verification methods
    Returns: dict with verified counts and bounding box information
    """
    # Check if scanned first
    is_scanned, scanned_details = check_if_scanned(file_path)
    
    # Configure Docling pipeline for accurate detection
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True # Critical for scanned docs
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
        mode="accurate"
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    start_time = time.time()
    result = converter.convert(file_path)
    processing_time = time.time() - start_time
    
    # Store bounding boxes for visualization
    # We will use a set to avoid duplicates if methods overlap
    image_bboxes = [] 
    table_bboxes = []
    
    # Method 1 & 3 Combined: Analyze the JSON structure recursively
    # This is often more reliable for finding nested items and bboxes than flat list iteration
    json_output = result.document.export_to_dict()
    
    found_images_json = 0
    found_tables_json = 0
    
    def extract_from_json(obj):
        nonlocal found_images_json, found_tables_json
        
        if isinstance(obj, dict):
            # Check for type
            item_type = obj.get('label', '').lower() # Docling sometimes uses label
            if not item_type:
                 # fallback to checking checks like self_ref
                 if 'self_ref' in obj and 'type' in obj: 
                     # This might be a flat item list style
                     pass
            
            # Identify by keys or explicit type fields common in Docling JSON
            # Note: Docling's export structure varies by version. 
            # We look for common markers for Pictures and Tables.
            
            is_picture = False
            is_table = False
            
            # Check 'type' if present (e.g. from flattened items)
            # Check 'label' (often 'picture', 'table')
            
            labels = [str(obj.get(k, '')).lower() for k in ['label', 'type', 'kind']]
            if any('picture' in l or 'image' in l for l in labels):
                is_picture = True
            elif any('table' in l for l in labels):
                is_table = True
                
            # Provenance / Location check
            # Docling JSON usually has a 'prov' list for location
            bboxes = []
            if 'prov' in obj and isinstance(obj['prov'], list):
                for p in obj['prov']:
                    if 'bbox' in p and 'page_no' in p:
                        bboxes.append((p['page_no'], p['bbox']))
            
            if is_picture:
                found_images_json += 1
                image_bboxes.extend(bboxes)
            elif is_table:
                found_tables_json += 1
                table_bboxes.extend(bboxes)
            
            # Recurse
            for k, v in obj.items():
                extract_from_json(v)
                
        elif isinstance(obj, list):
            for item in obj:
                extract_from_json(item)

    extract_from_json(json_output)
    
    # Method 2: Count from markdown output
    md_text = result.document.export_to_markdown()
    
    # Count image markers in markdown
    image_count_method2 = md_text.count('<!-- image -->')
    
    # Count tables in markdown
    table_count_method2 = 0
    lines = md_text.split('\n')
    for line in lines:
        stripped = line.strip()
        if('|' in stripped and '--' in stripped):
             if stripped.count('|') >= 3:
                table_count_method2 += 1
                
    # Use JSON counts (Method 3) as the "Structure" count now, as it's more exhaustive than the previous Method 1
    image_count_method1 = found_images_json
    table_count_method1 = found_tables_json
    
    # Double-check: Use the maximum count
    final_image_count = max(image_count_method1, image_count_method2)
    final_table_count = max(table_count_method1, table_count_method2)
    
    # Store all counts
    verification_data = {
        'images': {
            'method1_structure': image_count_method1,
            'method2_markdown': image_count_method2,
            'final': final_image_count,
            'bboxes': image_bboxes
        },
        'tables': {
            'method1_structure': table_count_method1,
            'method2_markdown': table_count_method2,
            'final': final_table_count,
            'bboxes': table_bboxes
        },
        'processing_time': processing_time,
        'scanned_status': {
            'is_scanned': is_scanned,
            'details': scanned_details
        }
    }
    
    return verification_data

def create_highlighted_pdf(pdf_path, image_bboxes, table_bboxes, output_path):
    """
    Create a new PDF with highlighted bounding boxes
    """
    doc = fitz.open(pdf_path)
    
    # Group bboxes by page
    page_images = {}
    page_tables = {}
    
    for page_num, bbox in image_bboxes:
        if page_num not in page_images:
            page_images[page_num] = []
        page_images[page_num].append(bbox)
    
    for page_num, bbox in table_bboxes:
        if page_num not in page_tables:
            page_tables[page_num] = []
        page_tables[page_num].append(bbox)
    
    count_drawn = 0
    
    # Draw rectangles on each page
    for page_idx in range(len(doc)):
        # Docling page numbers are 1-based usually
        # We need to match page_num (from Docling) to page_idx (0-based)
        # Assuming Docling 'page_no' 1 maps to doc[0]
        
        current_page_num = page_idx + 1
        page = doc[page_idx]
        page_height = page.rect.height
        
        # Draw image bboxes
        if current_page_num in page_images:
            for bbox in page_images[current_page_num]:
                try:
                    # Bbox can be dict or object depending on parsing
                    # Docling JSON bbox: {l, t, r, b, ...} or [l, t, r, b]?
                    # Normalized or absolute? Docling usually absolute bottom-left origin.
                    
                    x0, y0, x1, y1 = 0, 0, 0, 0
                    
                    if isinstance(bbox, dict):
                        x0 = bbox.get('l', 0)
                        # Flip Y: Docling is bottom-up? Check CoordOrigin.
                        # Usually: y_top = page_height - bbox.t
                        # But Docling V2 might use top-left origin?
                        # Standard Docling: Bottom-Left Origin.
                        # y0 (top of rect in PDF coords) = page_height - bbox['t'] (if t is top) 
                        # actually:
                        # l = left, b = bottom (y=0 is bottom), r = right, t = top
                        # PyMuPDF: Top-Left origin.
                        # So PyMuPDF y0 = page_height - bbox['t']
                        # PyMuPDF y1 = page_height - bbox['b']
                        
                        # Let's try standard Bottom-Left conversion first
                        t = bbox.get('t', 0)
                        b = bbox.get('b', 0)
                        
                        # Note: Docling 't' is usually larger than 'b' in bottom-left coords?
                        # No, usually t is y-coordinate of top edge (higher value), b is bottom (lower value).
                        
                        x0 = bbox.get('l', 0)
                        x1 = bbox.get('r', 0)
                        
                        # Convert to Top-Left system
                        rect_y0 = page_height - t
                        rect_y1 = page_height - b
                        
                        # Create rect
                        rect = fitz.Rect(x0, rect_y0, x1, rect_y1)
                        
                        page.draw_rect(rect, color=(0, 0, 1), width=3)
                        page.insert_text((x0, rect_y0 - 5), "IMAGE", fontsize=12, color=(0,0,1))
                        count_drawn += 1

                except Exception as e:
                    print(f"Error drawing bbox: {e}")

        # Draw table bboxes
        if current_page_num in page_tables:
            for bbox in page_tables[current_page_num]:
                try:
                    t = bbox.get('t', 0)
                    b = bbox.get('b', 0)
                    x0 = bbox.get('l', 0)
                    x1 = bbox.get('r', 0)
                    
                    rect_y0 = page_height - t
                    rect_y1 = page_height - b
                    
                    rect = fitz.Rect(x0, rect_y0, x1, rect_y1)
                    page.draw_rect(rect, color=(0, 0.7, 0), width=3)
                    page.insert_text((x0, rect_y0 - 5), "TABLE", fontsize=12, color=(0, 0.7, 0))
                    count_drawn += 1
                except:
                    pass

    doc.save(output_path)
    doc.close()
    return output_path, count_drawn

def pdf_to_images(pdf_path, dpi=150):
    """Convert PDF pages to images for display"""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append((page_num + 1, img))
    
    doc.close()
    return images

def main():
    st.title("🔢 PDF Image & Table Counter")
    st.markdown("""
    Upload PDF(s) to get accurate counts of **images** and **tables**.
    """)
    
    with st.sidebar:
        st.header("📂 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        st.header("⚙️ Settings")
        show_highlighted_pdf = st.checkbox("Show highlighted PDF", value=True)
    
    if not uploaded_files:
        st.info("👆 Upload a PDF file to start")
        return
    
    for uploaded_file in uploaded_files:
        st.divider()
        st.header(f"📄 {uploaded_file.name}")
        
        file_path = save_uploaded_file(uploaded_file)
        
        try:
            with st.spinner("🔍 Analyzing..."):
                results = count_images_and_tables(file_path)
            
            # Scanned Status Badge
            if results['scanned_status']['is_scanned']:
                st.warning(f"📷 **Scanned Document Detected**")
                st.caption(f"Details: {results['scanned_status']['details']}")
                st.caption("Note: Location / Bounding boxes might be approximate or missing for scanned elements.")
            
            # Counts
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🖼️ Images", results['images']['final'])
            with col2:
                st.metric("📋 Tables", results['tables']['final'])
            
            # Parse Bboxes for validity (count how many we actually have)
            valid_image_bboxes = len(results['images']['bboxes'])
            valid_table_bboxes = len(results['tables']['bboxes'])
            
            # Visualization
            if show_highlighted_pdf:
                if (valid_image_bboxes + valid_table_bboxes) > 0:
                    st.subheader("📍 Visual Detection")
                    
                    highlighted_pdf_path = file_path.replace('.pdf', '_highlighted.pdf')
                    try:
                        _, drawn_count = create_highlighted_pdf(
                            file_path, 
                            results['images']['bboxes'], 
                            results['tables']['bboxes'], 
                            highlighted_pdf_path
                        )
                        
                        if drawn_count == 0:
                             st.warning("⚠️ Elements were detected, but could not be drawn on the PDF (coordinate mismatch).")
                        else:
                            images = pdf_to_images(highlighted_pdf_path)
                            for p_num, img in images:
                                st.image(img, caption=f"Page {p_num}", use_container_width=True)
                                
                    except Exception as e:
                        st.error(f"Error creating visualization: {e}")
                else:
                    if (results['images']['final'] > 0 or results['tables']['final'] > 0):
                        st.warning("⚠️ Images/Tables were detected in text/content, but no location data (bounding boxes) was returned by the analyzer.")
                        st.info("This often happens with scanned documents where the entire page is treated as an image, or when elements are inferred from OCR text without specific coordinates.")
                    else:
                        st.info("No images or tables found to visualize.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            
        finally:
            if os.path.exists(file_path):
                try: # cleanup
                    os.unlink(file_path)
                except: pass

if __name__ == "__main__":
    main()
