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
from detectors.ai_content_detector import AIContentDetector

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
    ai_content_detector = AIContentDetector()
    
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
    
    # Run AI content detection (if text is available)
    ai_result = None
    if file_type == 'pdf' and pdf_doc:
        try:
            # Extract text for AI analysis
            extracted_text = metadata_result.get('raw_data', {}).get('extracted_text_full', '')
            if not extracted_text:
                # Fallback: extract text directly
                text_parts = []
                for page_num in range(min(len(pdf_doc), 3)):
                    try:
                        page = pdf_doc[page_num]
                        page_text = page.get_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception:
                        continue
                extracted_text = '\n\n'.join(text_parts)
            
            if extracted_text and len(extracted_text.strip()) > 50:
                ai_result = ai_content_detector.analyze(extracted_text)
        except Exception as e:
            # AI detection failed, continue without it
            ai_result = None
    
    # Run PDF-specific detectors
    all_results = {
        'metadata': metadata_result,
        'pixel': noise_result
    }
    
    if ai_result:
        all_results['ai_content'] = ai_result
    
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
        
        # Add AI content to results if detected
        if ai_result and ai_result.get('is_ai_generated'):
            if 'ai_content' not in all_results:
                all_results['ai_content'] = ai_result
    else:
        confidence_result = None
    
    # Check for correlation between low ELA risk and suspicious metadata
    ela_risk_low = not noise_result.get('flags') or len([f for f in noise_result.get('flags', []) if 'smoothing' in f.lower() or 'ela' in f.lower()]) == 0
    metadata_suspicious = len(metadata_result.get('flags', [])) > 0 or metadata_result.get('risk_score', 0) > 30
    correlation_flag = None
    if ela_risk_low and metadata_suspicious:
        correlation_flag = "⚠️ Correlation: Low image manipulation risk (ELA) but suspicious metadata detected. Metadata issues may indicate document forgery even without visible image tampering."
    
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
        'ai_content': ai_result,
        'correlation_flag': correlation_flag,
        'all_results': all_results if file_type == 'pdf' else {}
    }
    
    # Close PDF document if opened
    if pdf_doc:
        pdf_doc.close()
    
    return analysis


def generate_forensic_report(analysis):
    """
    Generate a comprehensive narrative forensic report.
    
    Args:
        analysis: Complete analysis dictionary
        
    Returns:
        str: Formatted narrative report
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("FORENSIC DOCUMENT ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Document: {analysis['filename']}")
    report_lines.append(f"Analysis Date: {analysis['timestamp']}")
    report_lines.append(f"File Type: {analysis['file_type'].upper()}")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Extract key data
    metadata = analysis['metadata']
    metadata_raw = metadata.get('raw_data', {})
    forgery_score = metadata.get('risk_score', 0)
    trust_score = metadata.get('trust_score', 0)
    confidence_data = analysis.get('confidence')
    ai_content = analysis.get('ai_content')
    
    # Overall Assessment
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Determine overall verdict
    if confidence_data:
        confidence_level = confidence_data.get('confidence_level', 'Unknown')
        confidence_score = confidence_data.get('confidence_score', 0)
    else:
        confidence_level = "Not Available"
        confidence_score = 0
    
    # Generate verdict statement
    if forgery_score >= 70 or trust_score < 40:
        verdict_strength = "strong evidence"
        verdict_conclusion = "this document is likely not an original, unaltered document"
    elif forgery_score >= 50 or trust_score < 60:
        verdict_strength = "significant technical indicators"
        verdict_conclusion = "this document may not be an original, unaltered document"
    elif forgery_score >= 30 or trust_score < 70:
        verdict_strength = "some technical indicators"
        verdict_conclusion = "this document requires further verification"
    else:
        verdict_strength = "minimal technical indicators"
        verdict_conclusion = "this document appears to be authentic"
    
    report_lines.append(f"Based on the comprehensive forensic analysis, there are {verdict_strength} that")
    report_lines.append(f"{verdict_conclusion}.")
    report_lines.append("")
    
    if confidence_data:
        report_lines.append(f"Confidence Level: {confidence_level} ({confidence_score:.0f}% confidence)")
    report_lines.append(f"Forgery Probability Score: {forgery_score}/100")
    report_lines.append(f"Trust Score: {trust_score}/100")
    report_lines.append("")
    
    # Key Red Flags Section
    red_flags = []
    
    # Software mismatch
    pdf_meta = metadata_raw.get('pdf_metadata', {})
    producer = pdf_meta.get('producer', '')
    creator = pdf_meta.get('creator', '')
    author = pdf_meta.get('author', '')
    
    # Check for consumer software with institutional content
    institutional_indicators = metadata_raw.get('institutional_indicators', [])
    source_mismatch = metadata_raw.get('source_mismatch', {})
    
    if source_mismatch or (institutional_indicators and producer):
        software_name = source_mismatch.get('software', producer or creator)
        if software_name:
            red_flags.append({
                'title': 'Software Mismatch',
                'detail': f'The document was produced using "{software_name}".',
                'explanation': 'Official government/institutional documents are typically generated by internal document management systems, not by consumer-grade third-party software or Word-to-PDF conversion tools.'
            })
    
    # Suspicious metadata
    if author:
        red_flags.append({
            'title': 'Suspicious Metadata - Author Field',
            'detail': f'The PDF author is listed as "{author}".',
            'explanation': 'Official UKVI/Home Office documents do not typically carry the name of an individual staff member in the metadata properties. This suggests manual document creation or modification.'
        })
    
    # Timeline anomalies
    timeline_anomaly = metadata_raw.get('timeline_anomaly', {})
    if timeline_anomaly:
        days_diff = timeline_anomaly.get('days_difference', 0)
        creation = timeline_anomaly.get('creation', '')
        modification = timeline_anomaly.get('modification', '')
        
        if days_diff:
            red_flags.append({
                'title': 'Modification Gap Anomaly',
                'detail': f'There is a {days_diff}-day anomaly between the original creation ({creation[:10] if len(creation) > 10 else creation}) and the modification ({modification[:10] if len(modification) > 10 else modification}).',
                'explanation': 'This strongly indicates the file was edited or modified years after it was first "created", which is highly unusual for official documents.'
            })
    
    # Creation date discrepancy
    creation_date = pdf_meta.get('creationDate', '')
    mod_date = pdf_meta.get('modDate', '')
    if creation_date and '202' in creation_date:
        # Extract year from creation date
        try:
            if creation_date.startswith('D:'):
                creation_date = creation_date[2:]
            if len(creation_date) >= 4:
                creation_year = creation_date[:4]
                # Check if content mentions different years (basic check)
                extracted_text = metadata_raw.get('extracted_text', '')
                if extracted_text and ('2024' in extracted_text or '2025' in extracted_text or '2026' in extracted_text):
                    content_years = []
                    for year in ['2024', '2025', '2026', '2027']:
                        if year in extracted_text:
                            content_years.append(year)
                    if content_years and creation_year < max(content_years):
                        red_flags.append({
                            'title': 'Creation Date Discrepancy',
                            'detail': f'The creation date is listed as {creation_year}, but the content of the document discusses events in {", ".join(content_years)}.',
                            'explanation': 'This temporal inconsistency suggests the document metadata was set incorrectly or the document was created/modified at a different time than claimed.'
                        })
        except:
            pass
    
    # AI indicators
    if ai_content and ai_content.get('is_ai_generated'):
        ai_confidence = ai_content.get('confidence', 0)
        red_flags.append({
            'title': 'AI-Generated Content Indicators',
            'detail': f'The tool detected a {ai_confidence:.0f}% confidence level for AI-generated content.',
            'explanation': 'This is particularly notable in formal documents, as AI-generated text often follows predictable linguistic patterns found in Large Language Models (LLMs). Official documents are typically written by human staff members.'
        })
    
    # Missing metadata
    completeness = metadata_raw.get('metadata_completeness', {})
    if not completeness.get('is_complete', True) and not institutional_indicators:
        missing = completeness.get('missing_fields', [])
        if missing:
            red_flags.append({
                'title': 'Missing Critical Metadata',
                'detail': f'Critical metadata fields are missing: {", ".join(missing).title()}.',
                'explanation': 'Official documents typically contain complete metadata including author and creator information. Missing metadata may indicate document manipulation or re-saving.'
            })
    
    # Suspicious software
    suspicious_software = metadata_raw.get('suspicious_software')
    if suspicious_software:
        red_flags.append({
            'title': 'Digital Manipulation Software Detected',
            'detail': f'Software used: {suspicious_software.title()}.',
            'explanation': 'The presence of image editing or manipulation software in document metadata strongly suggests the document was digitally altered or created using inappropriate tools for official documents.'
        })
    
    # Low trust score
    if trust_score < 50:
        red_flags.append({
            'title': 'Low Trust Score',
            'detail': f'The overall {trust_score}/100 Trust Score and {confidence_level if confidence_data else "analysis"} verdict suggest a high probability of forgery or significant tampering.',
            'explanation': 'A low trust score indicates multiple suspicious indicators were detected, including metadata issues, software mismatches, or manipulation evidence.'
        })
    
    # Write Red Flags Section
    if red_flags:
        report_lines.append("KEY RED FLAGS FROM ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        for i, flag in enumerate(red_flags, 1):
            report_lines.append(f"{i}. {flag['title']}")
            report_lines.append("")
            report_lines.append(f"   {flag['detail']}")
            report_lines.append("")
            report_lines.append(f"   Explanation: {flag['explanation']}")
            report_lines.append("")
    
    # Additional Findings
    report_lines.append("ADDITIONAL FINDINGS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Metadata details
    if pdf_meta:
        report_lines.append("Document Metadata:")
        if author:
            report_lines.append(f"  - Author: {author}")
        if creator:
            report_lines.append(f"  - Creator: {creator}")
        if producer:
            report_lines.append(f"  - Producer: {producer}")
        if creation_date:
            report_lines.append(f"  - Creation Date: {creation_date}")
        if mod_date:
            report_lines.append(f"  - Modification Date: {mod_date}")
        report_lines.append("")
    
    # Institutional indicators
    if institutional_indicators:
        report_lines.append(f"Institutional Indicators Found: {', '.join(institutional_indicators)}")
        report_lines.append("")
    
    # AI content details
    if ai_content and ai_content.get('is_ai_generated'):
        ai_metrics = ai_content.get('metrics', {})
        report_lines.append("AI Content Detection Details:")
        if ai_metrics.get('punctuation_diversity') is not None:
            report_lines.append(f"  - Punctuation Diversity: {ai_metrics['punctuation_diversity']:.3f}")
        if ai_metrics.get('word_entropy') is not None:
            report_lines.append(f"  - Word Entropy: {ai_metrics['word_entropy']:.2f}")
        if ai_metrics.get('paragraph_uniformity') is not None:
            report_lines.append(f"  - Paragraph Uniformity: {ai_metrics['paragraph_uniformity']:.2f}")
        if ai_metrics.get('ai_phrases_count', 0) > 0:
            report_lines.append(f"  - AI-Typical Phrases Detected: {ai_metrics['ai_phrases_count']}")
        report_lines.append("")
    
    # Summary for Case
    report_lines.append("SUMMARY FOR YOUR CASE")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    if forgery_score >= 50 or trust_score < 50:
        report_lines.append("While the content itself may sound plausible, the metadata and digital signature")
        report_lines.append("analysis indicate that the file was likely manually assembled, modified, or")
        report_lines.append("reconstructed.")
        report_lines.append("")
        report_lines.append("In a professional or legal context, a document with:")
        if confidence_data:
            report_lines.append(f"- A '{confidence_level}' verdict")
        report_lines.append(f"- A {trust_score}/100 Trust Score")
        if author:
            report_lines.append(f"- '{author}' as the metadata author")
        if producer:
            report_lines.append(f"- '{producer}' as the producer software")
        report_lines.append("")
        report_lines.append("would not be considered a verified original without additional authentication.")
    else:
        report_lines.append("The document shows minimal indicators of manipulation. However, it is")
        report_lines.append("recommended to verify authenticity through official channels when dealing")
        report_lines.append("with critical documents.")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def format_report_for_display(analysis):
    """
    Format the forensic report for better Streamlit markdown display.
    
    Args:
        analysis: Complete analysis dictionary
        
    Returns:
        str: Formatted markdown report
    """
    metadata = analysis['metadata']
    metadata_raw = metadata.get('raw_data', {})
    forgery_score = metadata.get('risk_score', 0)
    trust_score = metadata.get('trust_score', 0)
    confidence_data = analysis.get('confidence')
    ai_content = analysis.get('ai_content')
    
    report_parts = []
    
    # Header
    report_parts.append("## 📋 Forensic Document Analysis Report")
    report_parts.append("")
    report_parts.append(f"**Document:** {analysis['filename']}  ")
    report_parts.append(f"**Analysis Date:** {analysis['timestamp']}  ")
    report_parts.append(f"**File Type:** {analysis['file_type'].upper()}")
    report_parts.append("")
    report_parts.append("---")
    report_parts.append("")
    
    # Executive Summary
    report_parts.append("### Executive Summary")
    report_parts.append("")
    
    if confidence_data:
        confidence_level = confidence_data.get('confidence_level', 'Unknown')
        confidence_score = confidence_data.get('confidence_score', 0)
    else:
        confidence_level = "Not Available"
        confidence_score = 0
    
    # Generate verdict statement
    if forgery_score >= 70 or trust_score < 40:
        verdict_strength = "**strong evidence**"
        verdict_conclusion = "this document is **likely not an original, unaltered document**"
    elif forgery_score >= 50 or trust_score < 60:
        verdict_strength = "**significant technical indicators**"
        verdict_conclusion = "this document **may not be an original, unaltered document**"
    elif forgery_score >= 30 or trust_score < 70:
        verdict_strength = "**some technical indicators**"
        verdict_conclusion = "this document **requires further verification**"
    else:
        verdict_strength = "**minimal technical indicators**"
        verdict_conclusion = "this document **appears to be authentic**"
    
    report_parts.append(f"Based on the comprehensive forensic analysis, there are {verdict_strength} that")
    report_parts.append(f"{verdict_conclusion}.")
    report_parts.append("")
    
    # Key Metrics
    report_parts.append("**Key Metrics:**")
    if confidence_data:
        report_parts.append(f"- **Confidence Level:** {confidence_level} ({confidence_score:.0f}% confidence)")
    report_parts.append(f"- **Forgery Probability Score:** {forgery_score}/100")
    report_parts.append(f"- **Trust Score:** {trust_score}/100")
    report_parts.append("")
    
    # Key Red Flags
    red_flags = []
    pdf_meta = metadata_raw.get('pdf_metadata', {})
    producer = pdf_meta.get('producer', '')
    creator = pdf_meta.get('creator', '')
    author = pdf_meta.get('author', '')
    institutional_indicators = metadata_raw.get('institutional_indicators', [])
    source_mismatch = metadata_raw.get('source_mismatch', {})
    
    if source_mismatch or (institutional_indicators and producer):
        software_name = source_mismatch.get('software', producer or creator)
        if software_name:
            red_flags.append({
                'title': 'Software Mismatch',
                'detail': f'The document was produced using **"{software_name}"**.',
                'explanation': 'Official government/institutional documents are typically generated by internal document management systems, not by consumer-grade third-party software or Word-to-PDF conversion tools.'
            })
    
    if author:
        red_flags.append({
            'title': 'Suspicious Metadata - Author Field',
            'detail': f'The PDF author is listed as **"{author}"**.',
            'explanation': 'Official UKVI/Home Office documents do not typically carry the name of an individual staff member in the metadata properties. This suggests manual document creation or modification.'
        })
    
    timeline_anomaly = metadata_raw.get('timeline_anomaly', {})
    if timeline_anomaly:
        days_diff = timeline_anomaly.get('days_difference', 0)
        creation = timeline_anomaly.get('creation', '')
        modification = timeline_anomaly.get('modification', '')
        
        if days_diff:
            red_flags.append({
                'title': 'Modification Gap Anomaly',
                'detail': f'There is a **{days_diff}-day anomaly** between the original creation ({creation[:10] if len(creation) > 10 else creation}) and the modification ({modification[:10] if len(modification) > 10 else modification}).',
                'explanation': 'This strongly indicates the file was edited or modified years after it was first "created", which is highly unusual for official documents.'
            })
    
    creation_date = pdf_meta.get('creationDate', '')
    if creation_date and '202' in creation_date:
        try:
            if creation_date.startswith('D:'):
                creation_date = creation_date[2:]
            if len(creation_date) >= 4:
                creation_year = creation_date[:4]
                extracted_text = metadata_raw.get('extracted_text', '')
                if extracted_text and ('2024' in extracted_text or '2025' in extracted_text or '2026' in extracted_text):
                    content_years = []
                    for year in ['2024', '2025', '2026', '2027']:
                        if year in extracted_text:
                            content_years.append(year)
                    if content_years and creation_year < max(content_years):
                        red_flags.append({
                            'title': 'Creation Date Discrepancy',
                            'detail': f'The creation date is listed as **{creation_year}**, but the content of the document discusses events in **{", ".join(content_years)}**.',
                            'explanation': 'This temporal inconsistency suggests the document metadata was set incorrectly or the document was created/modified at a different time than claimed.'
                        })
        except:
            pass
    
    if ai_content and ai_content.get('is_ai_generated'):
        ai_confidence = ai_content.get('confidence', 0)
        red_flags.append({
            'title': 'AI-Generated Content Indicators',
            'detail': f'The tool detected a **{ai_confidence:.0f}% confidence level** for AI-generated content.',
            'explanation': 'This is particularly notable in formal documents, as AI-generated text often follows predictable linguistic patterns found in Large Language Models (LLMs). Official documents are typically written by human staff members.'
        })
    
    completeness = metadata_raw.get('metadata_completeness', {})
    if not completeness.get('is_complete', True) and not institutional_indicators:
        missing = completeness.get('missing_fields', [])
        if missing:
            red_flags.append({
                'title': 'Missing Critical Metadata',
                'detail': f'Critical metadata fields are missing: **{", ".join(missing).title()}**.',
                'explanation': 'Official documents typically contain complete metadata including author and creator information. Missing metadata may indicate document manipulation or re-saving.'
            })
    
    suspicious_software = metadata_raw.get('suspicious_software')
    if suspicious_software:
        red_flags.append({
            'title': 'Digital Manipulation Software Detected',
            'detail': f'Software used: **{suspicious_software.title()}**.',
            'explanation': 'The presence of image editing or manipulation software in document metadata strongly suggests the document was digitally altered or created using inappropriate tools for official documents.'
        })
    
    if trust_score < 50:
        red_flags.append({
            'title': 'Low Trust Score',
            'detail': f'The overall **{trust_score}/100 Trust Score** and **{confidence_level if confidence_data else "analysis"}** verdict suggest a high probability of forgery or significant tampering.',
            'explanation': 'A low trust score indicates multiple suspicious indicators were detected, including metadata issues, software mismatches, or manipulation evidence.'
        })
    
    if red_flags:
        report_parts.append("### 🔴 Key Red Flags from Analysis")
        report_parts.append("")
        
        for i, flag in enumerate(red_flags, 1):
            report_parts.append(f"#### {i}. {flag['title']}")
            report_parts.append("")
            report_parts.append(flag['detail'])
            report_parts.append("")
            report_parts.append(f"*Explanation:* {flag['explanation']}")
            report_parts.append("")
    
    # Summary
    report_parts.append("---")
    report_parts.append("")
    report_parts.append("### Summary for Your Case")
    report_parts.append("")
    
    if forgery_score >= 50 or trust_score < 50:
        report_parts.append("While the content itself may sound plausible, the metadata and digital signature")
        report_parts.append("analysis indicate that the file was likely manually assembled, modified, or")
        report_parts.append("reconstructed.")
        report_parts.append("")
        report_parts.append("In a professional or legal context, a document with:")
        report_parts.append("")
        if confidence_data:
            report_parts.append(f"- A **'{confidence_level}'** verdict")
        report_parts.append(f"- A **{trust_score}/100 Trust Score**")
        if author:
            report_parts.append(f"- **'{author}'** as the metadata author")
        if producer:
            report_parts.append(f"- **'{producer}'** as the producer software")
        report_parts.append("")
        report_parts.append("would **not be considered a verified original** without additional authentication.")
    else:
        report_parts.append("The document shows minimal indicators of manipulation. However, it is")
        report_parts.append("recommended to verify authenticity through official channels when dealing")
        report_parts.append("with critical documents.")
    
    return "\n".join(report_parts)


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
    
    # Create main tabs for Analysis Dashboard, Detailed Analysis, and Forensic Report
    main_tab1, main_tab2, main_tab3 = st.tabs(['📊 Analysis Dashboard', '🔍 Detailed Analysis', '📄 Forensic Report'])
    
    with main_tab1:
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
        with st.expander("ℹ️ Understanding Your Results", expanded=False):
            st.markdown("""
            **Forensic Verdict** provides a comprehensive assessment of document authenticity.
            
            **Key Components:**
            1. **Confidence Score**: How certain the system is about its assessment
            2. **Forgery Probability**: Likelihood the document was manipulated (0-100)
            3. **Trust Score**: How trustworthy the document metadata appears (0-100)
            4. **Forensic Signals**: Specific anomalies and issues detected
            5. **File DNA**: Extracted metadata and document properties
            
            **Interpreting Scores:**
            - **High Forgery Probability + Low Trust Score** = Strong evidence of fraud
            - **Low Forgery Probability + High Trust Score** = Document appears authentic
            - **Mixed scores** = Requires manual review and verification
            
            Always combine automated analysis with manual document review for critical decisions.
            """)
        
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
            with st.expander("ℹ️ What is Confidence Score?", expanded=False):
                st.markdown("""
                **Confidence Score** measures how certain the system is about its fraud detection assessment.
                
                - **90-100% (Definitive Fraud)**: Multiple strong indicators detected. High certainty of document manipulation.
                - **70-89% (High Suspicion)**: Several suspicious indicators found. Document likely fraudulent.
                - **50-69% (Moderate Suspicion)**: Some anomalies detected. Requires further investigation.
                - **0-49% (Low Suspicion)**: Few or weak indicators. Document appears legitimate.
                
                This score combines findings from metadata, structure, content, pixel, and signature analysis.
                """)
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
        
        st.markdown("#### Forgery Probability")
        with st.expander("ℹ️ What is Forgery Probability?", expanded=False):
            st.markdown("""
            **Forgery Probability** (0-100) indicates the likelihood that the document has been manipulated or forged.
            
            - **0-30 (Green)**: Low risk - Document appears authentic
            - **30-70 (Orange)**: Moderate risk - Some suspicious indicators found
            - **70-100 (Red)**: High risk - Strong evidence of manipulation or forgery
            
            This score is calculated from detected anomalies across all forensic analysis methods.
            """)
        forgery_gauge = create_forgery_gauge(forgery_score)
        st.plotly_chart(forgery_gauge, use_container_width=True)
        
        # Trust Score
        st.markdown("#### Trust Score")
        with st.expander("ℹ️ What is Trust Score?", expanded=False):
            st.markdown("""
            **Trust Score** (0-100) measures how trustworthy the document appears based on metadata and provenance.
            
            - **70-100 (High Trust)**: Document metadata is complete and consistent. Created with trusted software. No manipulation indicators.
            - **40-69 (Medium Trust)**: Some metadata missing or inconsistencies found. May be legitimate but requires verification.
            - **0-39 (Low Trust)**: Significant metadata issues, suspicious software detected, or manipulation indicators present.
            
            **Key Factors:**
            - Presence of author/creator metadata
            - Software used to create the document
            - Timeline consistency (creation vs modification dates)
            - Institutional indicators (for official documents)
            - Detection of manipulation software (Photoshop, etc.)
            
            A high trust score means the document's metadata suggests authenticity, while a low score indicates potential fraud.
            """)
        trust_color = "🟢" if trust_score >= 70 else "🟡" if trust_score >= 40 else "🔴"
        st.metric("Trust Score", f"{trust_score}/100", delta=None)
        st.caption(f"{trust_color} {'High Trust' if trust_score >= 70 else 'Medium Trust' if trust_score >= 40 else 'Low Trust'}")
        
        st.divider()
        
        # Summary of key findings (keep minimal on dashboard)
        st.markdown("### 🚨 Key Findings Summary")
        all_flags = []
        all_flags.extend(analysis['metadata']['flags'])
        all_flags.extend(analysis['noise']['flags'])
        
        # Add AI content detection flags if available
        if analysis.get('ai_content') and analysis['ai_content'].get('is_ai_generated'):
            ai_content = analysis['ai_content']
            confidence = ai_content.get('confidence', 0)
            all_flags.append(f"⚠️ AI-Generated Content Detected (Confidence: {confidence:.1f}%)")
        
        # Add correlation flag if detected
        if analysis.get('correlation_flag'):
            all_flags.append(analysis['correlation_flag'])
        
        # Show only top 5 most critical flags on dashboard
        critical_flags = [f for f in all_flags if any(keyword in f for keyword in ['Digital Manipulation', 'Time gap', 'AI-Generated', 'Missing Author', 'suspicious'])]
        if critical_flags:
            st.warning(f"**{len(critical_flags)} critical issue(s) detected.** See 'Detailed Analysis' tab for complete list.")
            for flag in critical_flags[:3]:  # Show top 3
                st.markdown(f"⚠️ {flag}")
            if len(critical_flags) > 3:
                st.caption(f"*+ {len(critical_flags) - 3} more issues. See Detailed Analysis tab.*")
        elif all_flags:
            st.info(f"**{len(all_flags)} indicator(s) found.** See 'Detailed Analysis' tab for details.")
        else:
            st.success("✅ No suspicious indicators detected")
    
    # Detailed Analysis Tab
    with main_tab2:
        st.markdown("## 🔍 Detailed Analysis")
        st.markdown("Complete forensic analysis results and detailed findings.")
        st.markdown("")
        
        # Forensic Signals
        st.markdown("### 🚨 Forensic Signals")
        with st.expander("ℹ️ What are Forensic Signals?", expanded=False):
            st.markdown("""
            **Forensic Signals** are specific anomalies and indicators detected during document analysis.
            
            **Signal Types:**
            - **🔴 Red Flags**: Strong evidence of manipulation (Photoshop detection, timeline anomalies)
            - **🟠 Orange Warnings**: Suspicious indicators requiring investigation
            - **🟡 Yellow Alerts**: Minor anomalies that may be legitimate
            - **ℹ️ Info**: Contextual information about the document
            
            **Common Signals:**
            - Time gap anomalies (document modified long after creation)
            - Suspicious software detection (Photoshop, GIMP, etc.)
            - Missing metadata (author, creator fields)
            - AI-generated content indicators
            - Pixel-level manipulation (ELA anomalies)
            - PDF structure inconsistencies
            
            Review each signal to understand what was detected and why it's significant.
            """)
        
        # Collect all flags
        all_flags = []
        all_flags.extend(analysis['metadata']['flags'])
        all_flags.extend(analysis['noise']['flags'])
        
        # Add AI content detection flags if available
        if analysis.get('ai_content') and analysis['ai_content'].get('is_ai_generated'):
            ai_content = analysis['ai_content']
            confidence = ai_content.get('confidence', 0)
            all_flags.append(f"⚠️ AI-Generated Content Detected (Confidence: {confidence:.1f}%)")
            # Add individual AI indicators
            for indicator in ai_content.get('indicators', []):
                all_flags.append(f"⚠️ {indicator}")
        
        # Add correlation flag if detected
        if analysis.get('correlation_flag'):
            all_flags.append(analysis['correlation_flag'])
        
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
        
        # AI Content Analysis (moved to Detailed Analysis tab)
        if analysis.get('ai_content') and analysis['ai_content'].get('is_ai_generated'):
            st.divider()
            st.markdown("### 🤖 AI Content Analysis")
            ai_content = analysis['ai_content']
            confidence = ai_content.get('confidence', 0)
            
            st.warning(f"**AI-Generated Content Detected** (Confidence: {confidence:.1f}%)")
            
            with st.expander("ℹ️ What is AI Content Detection?", expanded=False):
                st.markdown("""
                **AI Content Detection** analyzes text patterns to identify AI-generated content.
                
                **How it works:**
                - **Punctuation Diversity**: AI text often uses limited punctuation types
                - **Word Entropy**: Measures vocabulary variety (AI text tends to be less diverse)
                - **Paragraph Uniformity**: AI often produces paragraphs of similar length
                - **AI-Typical Phrases**: Detects common phrases used by AI writing assistants
                - **Sentence Variance**: AI text often has uniform sentence lengths
                - **Repetition Patterns**: Identifies repetitive word patterns common in AI text
                
                **Confidence Score**: Higher scores indicate stronger evidence of AI generation.
                """)
            
            metrics = ai_content.get('metrics', {})
            if metrics:
                with st.expander("📊 AI Detection Metrics", expanded=True):
                    if 'punctuation_diversity' in metrics:
                        st.metric("Punctuation Diversity", f"{metrics['punctuation_diversity']:.3f}")
                        st.caption("Lower values (< 0.05) suggest AI generation. Measures variety of punctuation marks used.")
                    
                    if 'word_entropy' in metrics:
                        st.metric("Word Entropy", f"{metrics['word_entropy']:.2f}")
                        st.caption("Lower values (< 8.5) suggest AI generation. Measures vocabulary diversity using Shannon entropy.")
                    
                    if 'paragraph_uniformity' in metrics:
                        st.metric("Paragraph Uniformity", f"{metrics['paragraph_uniformity']:.2f}")
                        st.caption("Higher values (> 0.85) suggest AI generation. Measures how similar paragraph lengths are.")
                    
                    if 'ai_phrases_count' in metrics:
                        st.metric("AI-Typical Phrases Found", metrics['ai_phrases_count'])
                        if metrics.get('ai_phrases'):
                            st.caption(f"Detected phrases: {', '.join(metrics['ai_phrases'][:10])}")
            
            # Show exact locations of AI phrases
            phrase_locations = metrics.get('ai_phrase_locations', [])
            if phrase_locations:
                st.markdown("#### 📍 Exact Locations of AI-Typical Phrases in Text")
                with st.expander("View detected phrases with exact text locations", expanded=True):
                    st.markdown("""
                    **How to read this:**
                    - Each detected AI-typical phrase is shown with its exact character position in the document
                    - The context shows surrounding text (50 characters before and after)
                    - The phrase is highlighted in **bold** within the context
                    - Use the character positions to locate these phrases in the original document
                    """)
                    
                    for i, loc in enumerate(phrase_locations[:15], 1):  # Show first 15
                        st.markdown(f"**{i}. Phrase: \"{loc['phrase']}\"**")
                        st.markdown(f"📍 **Position**: Characters {loc['position']} to {loc['end_position']} in document")
                        
                        # Show context with highlighted phrase
                        context = loc['context']
                        phrase_in_context = loc['phrase']
                        # Find the phrase in the context (case-insensitive)
                        context_lower = context.lower()
                        phrase_lower = phrase_in_context.lower()
                        phrase_start = context_lower.find(phrase_lower)
                        
                        if phrase_start != -1:
                            before = context[:phrase_start]
                            phrase_text = context[phrase_start:phrase_start + len(phrase_in_context)]
                            after = context[phrase_start + len(phrase_in_context):]
                            
                            # Create highlighted text
                            highlighted = f"...{before}**{phrase_text}**{after}..."
                            st.markdown(f"**Context (50 chars before/after):**")
                            st.markdown(highlighted)
                        else:
                            st.markdown(f"**Context:**")
                            st.code(context, language=None)
                        
                        if i < len(phrase_locations):
                            st.divider()
                    
                    if len(phrase_locations) > 15:
                        st.info(f"*Showing first 15 of {len(phrase_locations)} detected phrases. Scroll to see more.*")
            
            # Show other indicators
            indicators = ai_content.get('indicators', [])
            if indicators:
                st.markdown("#### 🔍 Additional AI Indicators")
                for indicator in indicators:
                    st.markdown(f"⚠️ {indicator}")
        
        st.divider()
        
        # File DNA
        st.markdown("### 🧬 File DNA")
        with st.expander("ℹ️ What is File DNA?", expanded=False):
            st.markdown("""
            **File DNA** contains the extracted metadata and properties that uniquely identify how the document was created.
            
            **Key Fields:**
            - **Author/Creator**: Who created the document (if available)
            - **Producer**: Software used to generate the PDF
            - **Creation Date**: When the document was first created
            - **Modification Date**: When the document was last modified
            - **Institutional Indicators**: Signs of official document origin
            
            **Why it matters:**
            - Legitimate documents typically have complete metadata
            - Official documents are usually created with specific software
            - Timeline consistency (creation vs modification) indicates authenticity
            - Missing metadata may indicate document manipulation or re-saving
            
            Compare this information with what you expect for a legitimate document.
            """)
        
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
    
    # Forensic Report Tab
    with main_tab3:
        st.markdown("## 📄 Comprehensive Forensic Report")
        st.markdown("")
        st.markdown("This report provides a comprehensive narrative summary of all forensic findings.")
        st.markdown("")
        
        # Generate and display the report
        report = generate_forensic_report(analysis)
        formatted_report = format_report_for_display(analysis)
        
        # Show formatted version (better for reading)
        st.markdown("### 📋 Report Summary")
        st.markdown(formatted_report)
        
        st.markdown("---")
        st.markdown("")
        
        # Show plain text version for download
        with st.expander("📄 Plain Text Version (for download)", expanded=False):
            st.markdown("```")
            st.text(report)
            st.markdown("```")
        
        # Download button
        st.download_button(
            label="📥 Download Report as Text File",
            data=report,
            file_name=f"forensic_report_{analysis['filename'].replace(' ', '_').replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_forensic_report"
        )
