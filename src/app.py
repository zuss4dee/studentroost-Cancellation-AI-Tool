"""
Main Streamlit Dashboard

Document Fraud Detection System - Main Application
"""

import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from detectors.metadata_detector import MetadataDetector
from detectors.pixel_detector import PixelDetector

# Page configuration
st.set_page_config(
    layout='wide',
    page_title='Document Fraud Detection',
    page_icon='🔍'
)

# Initialize session state
if 'recent_scans' not in st.session_state:
    st.session_state.recent_scans = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'ela_heatmap' not in st.session_state:
    st.session_state.ela_heatmap = None


def pdf_to_image(pdf_bytes):
    """Convert first page of PDF to PIL Image."""
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    first_page = pdf_doc[0]
    pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf_doc.close()
    return img


def get_file_type(filename):
    """Determine file type from filename."""
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        return 'pdf'
    elif ext in ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp']:
        return 'image'
    return 'unknown'


def analyze_file(uploaded_file):
    """Run all detectors on uploaded file."""
    file_bytes = uploaded_file.read()
    file_type = get_file_type(uploaded_file.name)
    
    # Store file for display
    file_stream = BytesIO(file_bytes)
    
    # Initialize detectors
    metadata_detector = MetadataDetector()
    pixel_detector = PixelDetector()
    
    # Run metadata analysis
    metadata_result = metadata_detector.analyze(BytesIO(file_bytes), file_type)
    
    # Prepare image for pixel analysis
    if file_type == 'pdf':
        display_image = pdf_to_image(file_bytes)
    else:
        file_stream.seek(0)
        display_image = Image.open(file_stream)
    
    # Run pixel analysis
    ela_heatmap = pixel_detector.analyze_ela(display_image)
    noise_result = pixel_detector.analyze_noise(display_image)
    
    # Adjust trust score based on pixel analysis findings
    # If pixel analysis detects manipulation, lower trust score
    if noise_result['flags']:  # If any flags from noise analysis (e.g., smoothing detected)
        # Lower trust score if pixel manipulation is detected
        if metadata_result['trust_score'] > 50:
            metadata_result['trust_score'] = 20  # Lower trust due to pixel-level manipulation
        else:
            metadata_result['trust_score'] = min(metadata_result['trust_score'], 20)
    
    # Combine results
    analysis = {
        'filename': uploaded_file.name,
        'file_type': file_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metadata': metadata_result,
        'noise': noise_result,
        'display_image': display_image,
        'ela_heatmap': ela_heatmap
    }
    
    return analysis


def create_forgery_gauge(forgery_score):
    """Create a color-coded gauge chart for forgery probability."""
    # Determine color based on forgery score
    if forgery_score < 30:
        color = 'green'
    elif forgery_score < 70:
        color = 'orange'
    else:
        color = 'red'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = forgery_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Forgery Probability"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig


# Sidebar
with st.sidebar:
    st.title('🔍 Case Files')
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'tif'],
        help="Upload a PDF or image file for analysis"
    )
    
    if uploaded_file is not None:
        # Analyze file
        with st.status("Scanning document layers...", expanded=True) as status:
            st.write("Extracting metadata...")
            analysis = analyze_file(uploaded_file)
            st.write("Performing pixel-level analysis...")
            st.write("Generating forensic report...")
            status.update(label="Analysis complete!", state="complete")
            
            st.session_state.current_file = uploaded_file.name
            st.session_state.current_analysis = analysis
            st.session_state.current_image = analysis['display_image']
            st.session_state.ela_heatmap = analysis['ela_heatmap']
            
            # Add to recent scans
            scan_entry = {
                'filename': analysis['filename'],
                'timestamp': analysis['timestamp'],
                'risk_score': analysis['metadata']['risk_score']
            }
            st.session_state.recent_scans.insert(0, scan_entry)
            # Keep only last 10 scans
            st.session_state.recent_scans = st.session_state.recent_scans[:10]
    
    # Recent scans
    st.divider()
    st.subheader('Recent Scans')
    
    if st.session_state.recent_scans:
        for i, scan in enumerate(st.session_state.recent_scans):
            risk_color = "🟢" if scan['risk_score'] < 30 else "🟡" if scan['risk_score'] < 70 else "🔴"
            st.write(f"{risk_color} **{scan['filename']}**")
            st.caption(f"Forgery Probability: {scan['risk_score']}/100 | {scan['timestamp']}")
            if i < len(st.session_state.recent_scans) - 1:
                st.divider()
    else:
        st.info('No scans yet. Upload a file to begin.')


# Main area
st.title('📄 Document Fraud Detection System')
st.caption('Forensic Analysis Dashboard')

if st.session_state.current_analysis is None:
    st.info('👈 Upload a document from the sidebar to begin analysis.')
else:
    analysis = st.session_state.current_analysis
    
    # Two-column layout
    col1, col2 = st.columns([1.2, 1])
    
    # Column 1: Viewer
    with col1:
        st.subheader('📋 Document Viewer')
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(['Original', 'ELA Heatmap', 'Metadata View'])
        
        with tab1:
            if analysis['file_type'] == 'pdf':
                st.image(analysis['display_image'], caption=analysis['filename'], use_container_width=True)
            else:
                st.image(analysis['display_image'], caption=analysis['filename'], use_container_width=True)
        
        with tab2:
            if st.session_state.ela_heatmap:
                # PDF-specific warning
                if analysis['file_type'] == 'pdf':
                    st.warning("""
                    ⚠️ **PDF Text Edit Limitation**: ELA analysis is most effective for raster image manipulation (JPEG, PNG). 
                    For PDFs, ELA analyzes the rendered image output, which may not detect text-only edits since PDFs are vector-based documents. 
                    **Metadata analysis is more reliable for detecting PDF text edits** - check the Forensic Signals and File DNA sections for editing software detection.
                    """)
                
                # Instructions section
                with st.expander("📖 How Error Level Analysis (ELA) Works", expanded=False):
                    st.markdown("""
                    **Error Level Analysis (ELA)** is a forensic technique used to detect digital image manipulation by identifying compression inconsistencies.
                    
                    **How it works:**
                    1. **Re-compression**: The original image is saved at 90% JPEG quality in memory
                    2. **Difference Calculation**: The system calculates the absolute difference between the original and re-compressed image
                    3. **Enhancement**: The difference is enhanced by 20x to make subtle variations visible
                    4. **Visualization**: The result is displayed as a heatmap where brightness indicates error levels
                    
                    **Interpreting the Heatmap:**
                    - **Dark regions** (low brightness): Areas with consistent compression - likely unedited
                    - **Bright regions** (high brightness): Areas with compression inconsistencies - potential signs of:
                      - Digital editing or manipulation
                      - Copy-paste operations
                      - Image compositing
                      - Selective quality adjustments
                      - Compression artifacts from editing tools
                    
                    **Important Notes:**
                    - **PDF Limitations**: ELA is less effective for text-only edits in PDFs. PDFs are vector-based, so text changes don't create the same compression artifacts as image edits. Metadata analysis (software detection, timestamps) is more reliable for PDF text manipulation.
                    - Bright areas don't always mean forgery - they can also indicate:
                      - Natural compression variations
                      - Text overlays or watermarks
                      - Different compression levels in source images
                    - Always combine ELA findings with metadata analysis for accurate assessment
                    - Professional forensic analysis requires multiple techniques
                    """)
                
                st.image(
                    st.session_state.ela_heatmap,
                    caption='Error Level Analysis (ELA) Heatmap - Bright regions indicate potential digital manipulation',
                    use_container_width=True
                )
                st.caption('💡 **Forensic Note:** Enhanced brightness regions in the ELA heatmap may indicate areas of digital alteration or compression artifacts.')
            else:
                st.info('ELA heatmap not available.')
        
        with tab3:
            st.subheader('Metadata Extraction')
            if analysis['metadata']['raw_data']:
                # Display metadata in expandable sections
                for key, value in analysis['metadata']['raw_data'].items():
                    if isinstance(value, dict):
                        with st.expander(f"📌 {key.replace('_', ' ').title()}"):
                            st.json(value)
                    else:
                        st.text(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info('No metadata extracted from document.')
    
    # Column 2: Verdict
    with col2:
        st.subheader('⚖️ Forensic Verdict')
        
        # Overall Forgery Probability
        forgery_score = analysis['metadata']['risk_score']
        trust_score = analysis['metadata']['trust_score']
        
        st.markdown("### Forgery Probability")
        forgery_gauge = create_forgery_gauge(forgery_score)
        st.plotly_chart(forgery_gauge, use_container_width=True)
        
        # Trust Score
        trust_color = "🟢" if trust_score >= 70 else "🟡" if trust_score >= 40 else "🔴"
        st.metric("Trust Score", f"{trust_score}/100", delta=None)
        st.caption(f"{trust_color} {'High Trust' if trust_score >= 70 else 'Medium Trust' if trust_score >= 40 else 'Low Trust'}")
        
        st.divider()
        
        # Forensic Signals
        st.markdown("### 🚨 Forensic Signals")
        
        all_flags = []
        all_flags.extend(analysis['metadata']['flags'])
        all_flags.extend(analysis['noise']['flags'])
        
        if all_flags:
            for flag in all_flags:
                if 'Digital Manipulation' in flag or 'Inconsistency' in flag:
                    st.markdown(f"🚩 **{flag}**")
                elif 'Potential' in flag or 'Smoothing' in flag:
                    st.markdown(f"⚠️ **{flag}**")
                else:
                    st.markdown(f"ℹ️ **{flag}**")
        else:
            st.success("✅ No suspicious forensic signals detected")
        
        st.divider()
        
        # File DNA
        st.markdown("### 🧬 File DNA")
        
        metadata = analysis['metadata']['raw_data']
        dna_data = []
        
        # Extract key metadata
        if analysis['file_type'] == 'pdf':
            pdf_meta = metadata.get('pdf_metadata', {})
            dna_data.append(['Author', pdf_meta.get('author', 'N/A')])
            dna_data.append(['Creator', pdf_meta.get('creator', 'N/A')])
            dna_data.append(['Producer', pdf_meta.get('producer', 'N/A')])
            dna_data.append(['Creation Date', pdf_meta.get('creationDate', 'N/A')])
            dna_data.append(['Modification Date', pdf_meta.get('modDate', 'N/A')])
        else:
            dna_data.append(['Software', metadata.get('software', 'N/A')])
            dna_data.append(['Camera Make', metadata.get('Make', 'N/A')])
            dna_data.append(['Camera Model', metadata.get('Model', 'N/A')])
            dna_data.append(['Date Taken', metadata.get('DateTime', 'N/A')])
        
        # Add noise variance
        dna_data.append(['Noise Variance', f"{analysis['noise']['variance']:.2f}"])
        
        if dna_data:
            df_dna = pd.DataFrame(dna_data, columns=['Property', 'Value'])
            st.dataframe(df_dna, use_container_width=True, hide_index=True)
        
        # Additional findings
        st.divider()
        st.markdown("### 📊 Forensic Analysis Details")
        
        with st.expander("Noise Variance Analysis"):
            st.write(analysis['noise']['findings'])
        
        with st.expander("Complete Metadata Extraction"):
            st.json(analysis['metadata']['raw_data'])
