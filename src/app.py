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
from detectors.pdf_structure_detector import PDFStructureDetector
from detectors.font_detector import FontDetector
from detectors.text_layer_detector import TextLayerDetector
from detectors.layout_detector import LayoutDetector
from detectors.signature_detector import SignatureDetector
from detectors.embedded_object_detector import EmbeddedObjectDetector
from detectors.confidence_scorer import ConfidenceScorer

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
    
    # Initialize all detectors
    metadata_detector = MetadataDetector()
    pixel_detector = PixelDetector()
    
    # For PDFs, initialize PDF-specific detectors
    pdf_doc = None
    if file_type == 'pdf':
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        structure_detector = PDFStructureDetector()
        font_detector = FontDetector()
        text_layer_detector = TextLayerDetector()
        layout_detector = LayoutDetector()
        signature_detector = SignatureDetector()
        embedded_detector = EmbeddedObjectDetector()
        confidence_scorer = ConfidenceScorer()
    
    # Run metadata analysis (pass filename for additional analysis)
    metadata_result = metadata_detector.analyze(BytesIO(file_bytes), file_type, uploaded_file.name)
    
    # Prepare image for pixel analysis
    if file_type == 'pdf':
        display_image = pdf_to_image(file_bytes)
    else:
        file_stream.seek(0)
        display_image = Image.open(file_stream)
    
    # Run pixel analysis
    ela_heatmap = pixel_detector.analyze_ela(display_image)
    noise_result = pixel_detector.analyze_noise(display_image)
    
    # Run PDF-specific detectors
    all_results = {
        'metadata': metadata_result,
        'pixel': noise_result
    }
    
    if file_type == 'pdf' and pdf_doc:
        # Run all PDF-specific detectors
        structure_result = structure_detector.analyze(pdf_doc)
        font_result = font_detector.analyze(pdf_doc)
        text_layer_result = text_layer_detector.analyze(pdf_doc)
        layout_result = layout_detector.analyze(pdf_doc)
        signature_result = signature_detector.analyze(pdf_doc)
        embedded_result = embedded_detector.analyze(pdf_doc)
        
        all_results.update({
            'structure': structure_result,
            'font': font_result,
            'text_layer': text_layer_result,
            'layout': layout_result,
            'signature': signature_result,
            'embedded': embedded_result
        })
        
        # Calculate confidence score
        confidence_result = confidence_scorer.calculate_confidence(all_results)
        
        # Adjust trust score based on confidence level to ensure consistency
        if confidence_result:
            confidence_score = confidence_result['confidence_score']
            confidence_level = confidence_result['confidence_level']
            
            # Override trust score based on confidence level
            if confidence_level == 'Definitive Fraud' and confidence_score >= 90:
                # High confidence in fraud = very low trust
                metadata_result['trust_score'] = 15
            elif confidence_level == 'High Suspicion' and confidence_score >= 70:
                # High suspicion = low trust
                metadata_result['trust_score'] = 25
            elif confidence_level == 'Moderate Suspicion' and confidence_score >= 50:
                # Moderate suspicion = moderate-low trust
                metadata_result['trust_score'] = 40
            elif confidence_level == 'Low Suspicion' and confidence_score < 30:
                # Low suspicion with low confidence = trust the metadata score
                # Don't override - keep metadata trust score
                pass
            # For Low Suspicion with higher scores (30-49), slightly reduce trust
            elif confidence_level == 'Low Suspicion' and confidence_score >= 30:
                # Minor anomalies but still legitimate
                if metadata_result['trust_score'] > 70:
                    metadata_result['trust_score'] = max(60, metadata_result['trust_score'] - 10)
        
        # Adjust trust score based on pixel analysis findings
        if noise_result['flags']:
            if metadata_result['trust_score'] > 50:
                metadata_result['trust_score'] = 20
            else:
                metadata_result['trust_score'] = min(metadata_result['trust_score'], 20)
    else:
        confidence_result = None
    
    # Combine results
    analysis = {
        'filename': uploaded_file.name,
        'file_type': file_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metadata': metadata_result,
        'noise': noise_result,
        'display_image': display_image,
        'ela_heatmap': ela_heatmap,
        'confidence': confidence_result,
        'all_results': all_results if file_type == 'pdf' else {}
    }
    
    # Close PDF document if opened
    if pdf_doc:
        pdf_doc.close()
    
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
            if uploaded_file.name.lower().endswith('.pdf'):
                st.write("Analyzing PDF structure...")
                st.write("Checking fonts and text layers...")
                st.write("Analyzing layout consistency...")
                st.write("Verifying digital signatures...")
                st.write("Checking embedded objects...")
            st.write("Performing pixel-level analysis...")
            st.write("Calculating confidence scores...")
            st.write("Generating forensic report...")
            analysis = analyze_file(uploaded_file)
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
        
        # Confidence Score (if available for PDFs)
        if analysis.get('confidence') and analysis['file_type'] == 'pdf':
            confidence_data = analysis['confidence']
            confidence_score = confidence_data['confidence_score']
            confidence_level = confidence_data['confidence_level']
            
            # Display confidence with color coding
            if confidence_score >= 90:
                conf_color = "🔴"
                conf_bg = "background-color: #ffebee; padding: 10px; border-radius: 5px;"
            elif confidence_score >= 70:
                conf_color = "🟠"
                conf_bg = "background-color: #fff3e0; padding: 10px; border-radius: 5px;"
            elif confidence_score >= 50:
                conf_color = "🟡"
                conf_bg = "background-color: #fffde7; padding: 10px; border-radius: 5px;"
            else:
                conf_color = "🟢"
                conf_bg = "background-color: #e8f5e9; padding: 10px; border-radius: 5px;"
            
            st.markdown(f"### 🎯 Confidence Score: {confidence_score:.0f}%")
            st.markdown(f"**{conf_color} {confidence_level}**")
            st.info(confidence_data['recommendation'])
            
            # Confidence breakdown
            with st.expander("📊 Confidence Breakdown", expanded=False):
                breakdown = confidence_data['indicator_breakdown']
                st.markdown(f"**Total Indicators:** {confidence_data['indicator_count']}")
                st.markdown(f"**Unique Categories:** {confidence_data['unique_categories']}")
                st.markdown("**By Category:**")
                st.markdown(f"- Metadata: {breakdown['metadata_indicators']}")
                st.markdown(f"- Structure: {breakdown['structure_indicators']}")
                st.markdown(f"- Content: {breakdown['content_indicators']}")
                st.markdown(f"- Pixel: {breakdown['pixel_indicators']}")
                st.markdown(f"- Signature: {breakdown['signature_indicators']}")
        
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
        
        # Add flags from PDF-specific detectors if available
        if analysis['file_type'] == 'pdf' and analysis.get('all_results'):
            all_results = analysis['all_results']
            if 'structure' in all_results:
                all_flags.extend(all_results['structure'].get('flags', []))
            if 'font' in all_results:
                all_flags.extend(all_results['font'].get('flags', []))
            if 'text_layer' in all_results:
                all_flags.extend(all_results['text_layer'].get('flags', []))
            if 'layout' in all_results:
                all_flags.extend(all_results['layout'].get('flags', []))
            if 'signature' in all_results:
                all_flags.extend(all_results['signature'].get('flags', []))
            if 'embedded' in all_results:
                all_flags.extend(all_results['embedded'].get('flags', []))
        
        if all_flags:
            for flag in all_flags:
                if 'Missing Metadata' in flag and 'Institutional Content' in flag:
                    st.warning(f"⚠️ **{flag}**")
                elif 'Missing Author/Creator Metadata' in flag and 'High Suspicion' in flag:
                    st.error(f"🔴 **{flag}**")
                elif 'Missing Metadata' in flag:
                    st.warning(f"⚠️ **{flag}**")
                elif 'Digital Manipulation' in flag or 'Inconsistency' in flag:
                    st.markdown(f"🚩 **{flag}**")
                elif 'Potential' in flag or 'Smoothing' in flag:
                    st.markdown(f"⚠️ **{flag}**")
                elif 'Institutional Indicators' in flag:
                    st.info(f"ℹ️ **{flag}**")
                else:
                    st.markdown(f"ℹ️ **{flag}**")
            
            # Show assessment if available
            metadata = analysis['metadata']['raw_data']
            assessment = metadata.get('assessment')
            if assessment:
                # Determine risk level based on assessment content
                if 'Likely legitimate' in assessment or 're-saved' in assessment.lower():
                    st.info(f"📋 **Assessment**: {assessment}")
                elif 'Requires Verification' in assessment or 'verify' in assessment.lower():
                    st.warning(f"📋 **Assessment**: {assessment}")
                else:
                    st.error(f"📋 **Assessment**: {assessment}")
            
            # Show Producer as alternative if metadata is missing
            completeness = metadata.get('metadata_completeness', {})
            if not completeness.get('is_complete', True) and completeness.get('has_producer'):
                producer_note = metadata.get('producer_note', '')
                if producer_note:
                    st.info(f"💡 **{producer_note}**")
                else:
                    producer_val = metadata.get('producer') or 'Unknown'
                    st.info(f"💡 **Alternative Indicator Available**: Producer field shows '{producer_val}'")
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
            # Handle empty strings - show 'N/A' instead of blank
            author = (pdf_meta.get('author') or '').strip() or 'N/A'
            creator = (pdf_meta.get('creator') or '').strip() or 'N/A'
            producer = (pdf_meta.get('producer') or '').strip() or 'N/A'
            creation_date = (pdf_meta.get('creationDate') or '').strip() or 'N/A'
            mod_date = (pdf_meta.get('modDate') or '').strip() or 'N/A'
            
            dna_data.append(['Author', author])
            dna_data.append(['Creator', creator])
            dna_data.append(['Producer', producer])
            dna_data.append(['Creation Date', creation_date])
            dna_data.append(['Modification Date', mod_date])
            
            # Add institutional indicators if found
            if metadata.get('institutional_indicators'):
                indicators_str = ', '.join(metadata['institutional_indicators'])
                dna_data.append(['Institutional Indicators', indicators_str])
        else:
            software = (metadata.get('software') or '').strip() or 'N/A'
            camera_make = (metadata.get('Make') or '').strip() or 'N/A'
            camera_model = (metadata.get('Model') or '').strip() or 'N/A'
            date_taken = (metadata.get('DateTime') or '').strip() or 'N/A'
            
            dna_data.append(['Software', software])
            dna_data.append(['Camera Make', camera_make])
            dna_data.append(['Camera Model', camera_model])
            dna_data.append(['Date Taken', date_taken])
        
        # Add noise variance
        dna_data.append(['Noise Variance', f"{analysis['noise']['variance']:.2f}"])
        
        if dna_data:
            df_dna = pd.DataFrame(dna_data, columns=['Property', 'Value'])
            st.dataframe(df_dna, use_container_width=True, hide_index=True)
        
        # Alternative Analysis Section
        if analysis['file_type'] == 'pdf':
            metadata_completeness = metadata.get('metadata_completeness', {})
            extracted_text = metadata.get('extracted_text')
            institutional_indicators = metadata.get('institutional_indicators', [])
            filename_analysis = metadata.get('filename_analysis', [])
            
            if not metadata_completeness.get('is_complete', True) or extracted_text or institutional_indicators or filename_analysis:
                st.divider()
                st.markdown("### 🔍 Alternative Analysis")
                
                # Metadata Completeness Status
                if not metadata_completeness.get('is_complete', True):
                    missing = metadata_completeness.get('missing_fields', [])
                    st.warning(f"⚠️ **Metadata Incomplete**: Missing fields: {', '.join(missing).title()}")
                    
                    # Show assessment if available
                    assessment = metadata.get('assessment')
                    if assessment:
                        st.info(f"📋 **Assessment**: {assessment}")
                    
                    # Show producer information
                    if metadata_completeness.get('has_producer'):
                        producer_val = metadata.get('producer') or 'Unknown'
                        producer_note = metadata.get('producer_note', f"Producer: {producer_val}")
                        st.info(f"ℹ️ **{producer_note}**")
                        
                        # Show producer assessment if available
                        producer_assessment = metadata.get('producer_assessment')
                        if producer_assessment:
                            st.caption(f"💡 {producer_assessment}")
                
                # Filename Analysis
                if filename_analysis:
                    st.markdown("**Filename Analysis:**")
                    for indicator in filename_analysis:
                        st.markdown(f"- {indicator}")
                
                # Institutional Indicators
                if institutional_indicators:
                    st.markdown("**Institutional Indicators Found:**")
                    for indicator in institutional_indicators:
                        st.markdown(f"- {indicator.title()}")
                
                # Extracted Text Preview
                if extracted_text:
                    with st.expander("📄 Extracted Text Preview (First 200 characters)"):
                        preview = metadata.get('extracted_text_full', extracted_text)[:200]
                        st.text(preview)
                        if len(metadata.get('extracted_text_full', extracted_text)) > 200:
                            st.caption(f"Full text available ({len(metadata.get('extracted_text_full', extracted_text))} characters total)")
                            with st.expander("View Full Extracted Text"):
                                st.text(metadata.get('extracted_text_full', extracted_text))
        
        # Additional findings
        st.divider()
        st.markdown("### 📊 Forensic Analysis Details")
        
        # Show detailed findings from all detectors
        if analysis['file_type'] == 'pdf' and analysis.get('all_results'):
            all_results = analysis['all_results']
            
            with st.expander("🔍 PDF Structure Analysis", expanded=False):
                if 'structure' in all_results:
                    structure = all_results['structure']
                    if structure.get('findings'):
                        st.json(structure['findings'])
                    st.caption(f"Risk Score: {structure.get('risk_score', 0)}/100")
                else:
                    st.info("Structure analysis not available")
            
            with st.expander("🔤 Font Analysis", expanded=False):
                if 'font' in all_results:
                    font = all_results['font']
                    if font.get('findings'):
                        st.json(font['findings'])
                    st.caption(f"Risk Score: {font.get('risk_score', 0)}/100")
                else:
                    st.info("Font analysis not available")
            
            with st.expander("📝 Text Layer Analysis", expanded=False):
                if 'text_layer' in all_results:
                    text_layer = all_results['text_layer']
                    if text_layer.get('findings'):
                        st.json(text_layer['findings'])
                    st.caption(f"Risk Score: {text_layer.get('risk_score', 0)}/100")
                else:
                    st.info("Text layer analysis not available")
            
            with st.expander("📐 Layout Analysis", expanded=False):
                if 'layout' in all_results:
                    layout = all_results['layout']
                    if layout.get('findings'):
                        st.json(layout['findings'])
                    st.caption(f"Risk Score: {layout.get('risk_score', 0)}/100")
                else:
                    st.info("Layout analysis not available")
            
            with st.expander("✍️ Signature Analysis", expanded=False):
                if 'signature' in all_results:
                    signature = all_results['signature']
                    if signature.get('findings'):
                        st.json(signature['findings'])
                    st.caption(f"Risk Score: {signature.get('risk_score', 0)}/100")
                else:
                    st.info("Signature analysis not available")
            
            with st.expander("🖼️ Embedded Object Analysis", expanded=False):
                if 'embedded' in all_results:
                    embedded = all_results['embedded']
                    if embedded.get('findings'):
                        st.json(embedded['findings'])
                    st.caption(f"Risk Score: {embedded.get('risk_score', 0)}/100")
                else:
                    st.info("Embedded object analysis not available")
        
        with st.expander("🔊 Noise Variance Analysis"):
            st.write(analysis['noise']['findings'])
        
        with st.expander("📋 Complete Metadata Extraction"):
            st.json(analysis['metadata']['raw_data'])
