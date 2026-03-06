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
from policy_engine import PolicyEngine
from streamlit.components.v1 import html as st_html

# Page configuration
st.set_page_config(
    layout='wide',
    page_title='Student Roost | Fraud Ops Dashboard',
    page_icon='🔍'
)

# Global CSS for modern enterprise dashboard styling (injected via components to avoid rendering as text)
_GLOBAL_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  /* Hide default Streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  /* Global background & typography */
  html, body, [class*="css"] {
    font-family: 'Inter', 'Roboto', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    background-color: #F9FAFB !important;
  }

  /* Main content container */
  .main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    background-color: #F9FAFB;
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB;
  }
  section[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem;
  }

  /* Card utility class */
  .stCard {
    background-color: #FFFFFF;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    padding: 20px;
    border: 1px solid #E5E7EB;
  }

  /* Verdict badge & checklist styling */
  .verdict-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .verdict-badge.red   { background-color:#FEE2E2; color:#B91C1C; }
  .verdict-badge.amber { background-color:#FEF3C7; color:#92400E; }
  .verdict-badge.green { background-color:#DCFCE7; color:#166534; }

  .checklist-item {
    display: flex;
    gap: 0.5rem;
    font-size: 0.9rem;
    margin-top: 0.35rem;
  }
  .checklist-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    margin-top: 0.4rem;
    background: #D1D5DB;
  }

  /* Recent scans styled as nav links */
  .recent-scan-item {
    padding: 0.4rem 0.15rem;
    font-size: 0.9rem;
    color: #6B7280;
  }
  .recent-scan-item strong {
    color: #111827;
  }
  .recent-scan-meta {
    font-size: 0.75rem;
    color: #9CA3AF;
  }
</style>
"""

st_html(_GLOBAL_CSS, height=0)

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
if 'current_policy_result' not in st.session_state:
    st.session_state.current_policy_result = None
if 'current_doc_type_label' not in st.session_state:
    st.session_state.current_doc_type_label = "Unknown Type"


@st.cache_resource
def get_policy_engine():
    return PolicyEngine()


policy_engine = get_policy_engine()


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


def generate_forensic_report(analysis, policy_result=None, doc_type_label=None):
    """
    Generate a comprehensive narrative forensic report.
    Now supports Context-Aware Policy verdicts.
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
    if doc_type_label:
        report_lines.append(f"Document Class: {doc_type_label}")
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
    
    # Use Policy Engine Verdict if available
    if policy_result:
        verdict = policy_result['verdict']
        reason = policy_result['reason']
        action = policy_result['action']
        
        report_lines.append(f"POLICY APPLIED: {doc_type_label or 'Generic'}")
        report_lines.append(f"AUTOMATED VERDICT: {verdict}")
        report_lines.append(f"PRIMARY REASON: {reason}")
        report_lines.append("")
        
        if verdict == 'RED':
            report_lines.append("CONCLUSION: This document FAILS the specific validation policy for this")
            report_lines.append("document type. It should NOT be accepted without verification.")
            report_lines.append(f"Recommended Action: {action}")
        elif verdict == 'AMBER':
            report_lines.append("CONCLUSION: This document contains irregularities requiring MANUAL REVIEW.")
            report_lines.append(f"Recommended Action: {action}")
        else:
            report_lines.append("CONCLUSION: This document PASSES the forensic checks for this document type.")
            report_lines.append(f"Recommended Action: {action}")
        report_lines.append("")
    else:
        # Fallback to old generic logic
        if forgery_score >= 70 or trust_score < 40:
            verdict_conclusion = "this document is likely not an original, unaltered document"
        elif forgery_score >= 50 or trust_score < 60:
            verdict_conclusion = "this document may not be an original, unaltered document"
        elif forgery_score >= 30 or trust_score < 70:
            verdict_conclusion = "this document requires further verification"
        else:
            verdict_conclusion = "this document appears to be authentic"
        
        report_lines.append(f"Based on the comprehensive forensic analysis, {verdict_conclusion}.")
        report_lines.append("")
    
    # Metrics (ensure confidence_level/score set for Red Flags section below)
    confidence_level = "Not Available"
    confidence_score = 0
    if confidence_data:
        confidence_level = confidence_data.get('confidence_level', 'Unknown')
        confidence_score = confidence_data.get('confidence_score', 0)
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


def format_report_for_display(analysis, policy_result=None, doc_type_label=None):
    """
    Format the forensic report for better Streamlit markdown display.
    Supports context-aware policy verdicts when policy_result is provided.
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
    if doc_type_label:
        report_parts.append(f"**Document Class:** {doc_type_label}")
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
    
    if policy_result:
        verdict = policy_result.get('verdict', '')
        reason = policy_result.get('reason', '')
        action = policy_result.get('action', '')
        report_parts.append(f"**Policy Applied:** {doc_type_label or 'Generic'}")
        report_parts.append(f"**Verdict:** {verdict}")
        report_parts.append(f"**Reason:** {reason}")
        report_parts.append(f"**Recommended Action:** {action}")
        report_parts.append("")
    else:
        # Generate verdict statement (generic)
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
    # Workspace header
    st.markdown("**❖ Student Roost | Fraud Ops**")
    st.caption("Internal case review workspace")
    st.markdown("---")

    # Document Type Selector
    doc_type_label = st.selectbox(
        "Document Type",
        options=["Visa / UKVI Refusal", "University Letter", "Medical Note", "Other"],
        index=0,
        help="Select the type of document to apply specific forensic rules."
    )

    # Map label to config key
    type_map = {
        "Visa / UKVI Refusal": "visa_refusal",
        "University Letter": "university_letter",
        "Medical Note": "medical_letter",
        "Other": "generic"
    }
    selected_doc_type = type_map[doc_type_label]

    # File upload – simplified, clean label
    st.markdown("#### Upload document")
    uploaded_file = st.file_uploader(
        "Upload document",
        type=['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'tif'],
        help="Upload a PDF or image file for analysis"
    )

    if uploaded_file is not None:
        # Analyze file (logic unchanged)
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

            st.write("Applying business policies...")
            policy_result = policy_engine.evaluate(analysis, selected_doc_type)

            st.session_state.current_policy_result = policy_result
            st.session_state.current_doc_type_label = doc_type_label

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
    
    # Recent scans styled as nav items
    st.markdown("#### Recent scans")
    
    if st.session_state.recent_scans:
        for scan in st.session_state.recent_scans:
            risk_color = "🟢" if scan['risk_score'] < 30 else "🟡" if scan['risk_score'] < 70 else "🔴"
            st.markdown(
                f"""
                <div class="recent-scan-item">
                    {risk_color} <strong>{scan['filename']}</strong>
                    <div class="recent-scan-meta">Forgery {scan['risk_score']}/100 · {scan['timestamp']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption('No scans yet. Upload a file to begin.')


# Main area
st.markdown("# Case File Analysis 👋")
st.caption("Reviewing uploaded documents for forensic anomalies.")

if st.session_state.current_analysis is None:
    st.info('👈 Upload a document from the sidebar to begin analysis.')
else:
    analysis = st.session_state.current_analysis
    policy_result = getattr(st.session_state, 'current_policy_result', None)
    doc_type_label = getattr(st.session_state, 'current_doc_type_label', "Unknown Type")

    # Create main tabs for Analysis Dashboard, Detailed Analysis, and Forensic Report
    main_tab1, main_tab2, main_tab3 = st.tabs(['📊 Analysis Dashboard', '🔍 Detailed Analysis', '📄 Forensic Report'])
    
    # -------------------------------------------------------------------------
    # 📊 Analysis Dashboard – minimal triage view for internal + external users
    # -------------------------------------------------------------------------
    with main_tab1:
        # Top triage strip: verdict + key scores
        forgery_score = analysis['metadata']['risk_score']
        trust_score = analysis['metadata']['trust_score']
        confidence_data = analysis.get('confidence') if analysis.get('confidence') and analysis['file_type'] == 'pdf' else None

        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        top_left, top_mid, top_right = st.columns([2, 1, 1])

        with top_left:
            st.subheader("⚖️ Forensic verdict")

            if policy_result:
                verdict = policy_result['verdict']
                action = policy_result['action']
                reason = policy_result['reason']
                badge_class = "red" if verdict == "RED" else "amber" if verdict == "AMBER" else "green"

                st.markdown(
                    f"""
                    <span class="verdict-badge {badge_class}">
                        {verdict} · {action}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Reason:** {reason}")
            else:
                # Generic fallback if no policy result
                verdict_text = "Document appears authentic"
                if forgery_score >= 70 or trust_score < 40:
                    verdict_text = "High risk of forgery"
                elif forgery_score >= 40 or trust_score < 60:
                    verdict_text = "Moderate anomalies detected"
                st.markdown(f"**Verdict:** {verdict_text}")

            if confidence_data:
                st.caption(f"Confidence: {confidence_data['confidence_score']:.0f}% · {confidence_data['confidence_level']}")

        with top_mid:
            st.metric("Forgery score", f"{forgery_score}/100")

        with top_right:
            st.metric("Trust score", f"{trust_score}/100")

        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")

        # Second row: viewer (left) + key findings (right)
        col1, col2 = st.columns([1.4, 1])

        # Left: minimal document viewer
        with col1:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("📋 Document viewer")

            viewer_tab1, viewer_tab2 = st.tabs(["Original", "ELA heatmap"])

            with viewer_tab1:
                st.image(
                    analysis['display_image'],
                    caption=analysis['filename'],
                    use_container_width=True,
                )

            with viewer_tab2:
                if st.session_state.ela_heatmap is not None:
                    st.image(
                        st.session_state.ela_heatmap,
                        caption="Error Level Analysis (ELA) heatmap",
                        use_container_width=True,
                    )
                    st.caption("Bright regions may indicate digital manipulation. For details, see the Detailed Analysis tab.")
                else:
                    st.info("ELA heatmap not available for this document.")

            st.markdown('</div>', unsafe_allow_html=True)

        # Right: short key‑findings summary
        with col2:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("🚨 Key findings")

            all_flags = []
            all_flags.extend(analysis['metadata']['flags'])
            all_flags.extend(analysis['noise']['flags'])

            # AI content flag (if present)
            if analysis.get('ai_content') and analysis['ai_content'].get('is_ai_generated'):
                ai_content = analysis['ai_content']
                ai_conf = ai_content.get('confidence', 0)
                all_flags.append(f"AI‑generated content detected (confidence {ai_conf:.1f}%)")

            # Correlation flag (if present)
            if analysis.get('correlation_flag'):
                all_flags.append(analysis['correlation_flag'])

            # Filter to most critical issues
            critical_keywords = ["Digital Manipulation", "Time gap", "AI-Generated", "Missing Author", "suspicious"]
            critical_flags = [
                f for f in all_flags
                if any(k.lower() in f.lower() for k in critical_keywords)
            ]
            display_flags = critical_flags or all_flags

            if display_flags:
                for flag in display_flags[:3]:
                    st.markdown(f"- {flag}")

                remaining = max(0, len(display_flags) - 3)
                if remaining > 0:
                    st.caption(f"+ {remaining} more signals. See the Detailed Analysis tab for full list.")
            else:
                st.success("No suspicious findings in top‑level checks.")

            st.markdown('</div>', unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # 🔍 Detailed Analysis – second‑level drill‑down
    # -------------------------------------------------------------------------
    with main_tab2:
        st.markdown("## 🔍 Detailed analysis")
        st.caption("Deeper forensic signals, AI patterns, metadata, and detector outputs.")
        st.markdown("")

        # Sub-tabs to keep this view navigable
        sig_tab, ai_tab, dna_tab, det_tab = st.tabs(
            ["Forensic signals", "AI content", "File DNA & metadata", "Detector outputs"]
        )

        metadata = analysis['metadata']['raw_data']
        
        # --- Forensic signals (all flags in one place) ---
        with sig_tab:
            st.markdown("### 🚨 Forensic signals")

            all_flags = []
            all_flags.extend(analysis['metadata']['flags'])
            all_flags.extend(analysis['noise']['flags'])

            # AI‑related flags
            if analysis.get('ai_content') and analysis['ai_content'].get('is_ai_generated'):
                ai_content = analysis['ai_content']
                ai_conf = ai_content.get('confidence', 0)
                all_flags.append(f"⚠️ AI‑generated content detected (confidence {ai_conf:.1f}%)")
                for indicator in ai_content.get('indicators', []):
                    all_flags.append(f"⚠️ {indicator}")

            # Correlation flag
            if analysis.get('correlation_flag'):
                all_flags.append(analysis['correlation_flag'])

            # Flags from PDF‑specific detectors
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

                # High‑level assessment from metadata, if present
                assessment = metadata.get('assessment')
                if assessment:
                    if 'Likely legitimate' in assessment or 're-saved' in assessment.lower():
                        st.info(f"📋 **Assessment**: {assessment}")
                    elif 'Requires Verification' in assessment or 'verify' in assessment.lower():
                        st.warning(f"📋 **Assessment**: {assessment}")
                    else:
                        st.error(f"📋 **Assessment**: {assessment}")

                completeness = metadata.get('metadata_completeness', {})
                if not completeness.get('is_complete', True) and completeness.get('has_producer'):
                    producer_note = metadata.get('producer_note', '')
                    if producer_note:
                        st.info(f"💡 **{producer_note}**")
                    else:
                        producer_val = metadata.get('producer') or 'Unknown'
                        st.info(f"💡 **Alternative indicator**: producer field shows '{producer_val}'")
            else:
                st.success("✅ No suspicious forensic signals detected.")

        # --- AI content details ---
        with ai_tab:
            st.markdown("### 🤖 AI content analysis")

            if analysis.get('ai_content') and analysis['ai_content'].get('is_ai_generated'):
                ai_content = analysis['ai_content']
                confidence = ai_content.get('confidence', 0)

                st.warning(f"AI‑generated content detected (confidence {confidence:.1f}%)")

                metrics = ai_content.get('metrics', {})
                if metrics:
                    cols = st.columns(3)
                    if 'punctuation_diversity' in metrics:
                        cols[0].metric("Punctuation diversity", f"{metrics['punctuation_diversity']:.3f}")
                    if 'word_entropy' in metrics:
                        cols[1].metric("Word entropy", f"{metrics['word_entropy']:.2f}")
                    if 'paragraph_uniformity' in metrics:
                        cols[2].metric("Paragraph uniformity", f"{metrics['paragraph_uniformity']:.2f}")

                    if 'ai_phrases_count' in metrics:
                        st.caption(f"AI‑typical phrases found: {metrics['ai_phrases_count']}")

                # Exact locations for AI phrases (if available)
                phrase_locations = metrics.get('ai_phrase_locations', [])
                if phrase_locations:
                    with st.expander("Detected AI‑typical phrases with context", expanded=False):
                        for i, loc in enumerate(phrase_locations[:15], 1):
                            st.markdown(f"**{i}. \"{loc['phrase']}\"**")
                            st.markdown(f"Characters {loc['position']}–{loc['end_position']}")

                            context = loc['context']
                            phrase_in_context = loc['phrase']
                            context_lower = context.lower()
                            phrase_lower = phrase_in_context.lower()
                            phrase_start = context_lower.find(phrase_lower)

                            if phrase_start != -1:
                                before = context[:phrase_start]
                                phrase_text = context[phrase_start:phrase_start + len(phrase_in_context)]
                                after = context[phrase_start + len(phrase_in_context):]
                                highlighted = f"...{before}**{phrase_text}**{after}..."
                                st.markdown(highlighted)
                            else:
                                st.code(context, language=None)

                            if i < len(phrase_locations):
                                st.divider()

                        if len(phrase_locations) > 15:
                            st.caption(f"Showing first 15 of {len(phrase_locations)} detected phrases.")

                indicators = ai_content.get('indicators', [])
                if indicators:
                    st.markdown("#### Additional AI indicators")
                    for indicator in indicators:
                        st.markdown(f"- {indicator}")
            else:
                st.info("No strong AI‑generated content indicators detected for this document.")

        # --- File DNA & metadata ---
        with dna_tab:
            st.markdown("### 🧬 File DNA & metadata")

            dna_data = []
        
        if analysis['file_type'] == 'pdf':
            pdf_meta = metadata.get('pdf_metadata', {})
            author = (pdf_meta.get('author') or '').strip() or 'N/A'
            creator = (pdf_meta.get('creator') or '').strip() or 'N/A'
            producer = (pdf_meta.get('producer') or '').strip() or 'N/A'
            creation_date = (pdf_meta.get('creationDate') or '').strip() or 'N/A'
            mod_date = (pdf_meta.get('modDate') or '').strip() or 'N/A'

            dna_data.append(['Author', author])
            dna_data.append(['Creator', creator])
            dna_data.append(['Producer', producer])
            dna_data.append(['Creation date', creation_date])
            dna_data.append(['Modification date', mod_date])

            if metadata.get('institutional_indicators'):
                indicators_str = ', '.join(metadata['institutional_indicators'])
                dna_data.append(['Institutional indicators', indicators_str])
        else:
            software = (metadata.get('software') or '').strip() or 'N/A'
            camera_make = (metadata.get('Make') or '').strip() or 'N/A'
            camera_model = (metadata.get('Model') or '').strip() or 'N/A'
            date_taken = (metadata.get('DateTime') or '').strip() or 'N/A'

            dna_data.append(['Software', software])
            dna_data.append(['Camera make', camera_make])
            dna_data.append(['Camera model', camera_model])
            dna_data.append(['Date taken', date_taken])

        dna_data.append(['Noise variance', f"{analysis['noise']['variance']:.2f}"])

        if dna_data:
            df_dna = pd.DataFrame(dna_data, columns=['Property', 'Value'])
            st.dataframe(df_dna, use_container_width=True, hide_index=True)

        # Alternative analysis (PDF only)
        if analysis['file_type'] == 'pdf':
            metadata_completeness = metadata.get('metadata_completeness', {})
            extracted_text = metadata.get('extracted_text')
            institutional_indicators = metadata.get('institutional_indicators', [])
            filename_analysis = metadata.get('filename_analysis', [])

            if (
                not metadata_completeness.get('is_complete', True)
                or extracted_text
                or institutional_indicators
                or filename_analysis
            ):
                st.markdown("#### Alternative indicators")

                if not metadata_completeness.get('is_complete', True):
                    missing = metadata_completeness.get('missing_fields', [])
                    st.warning(f"Metadata incomplete: missing {', '.join(missing).title()}")

                    if metadata_completeness.get('has_producer'):
                        producer_val = metadata.get('producer') or 'Unknown'
                        producer_note = metadata.get('producer_note', f"Producer: {producer_val}")
                        st.info(producer_note)

                        producer_assessment = metadata.get('producer_assessment')
                        if producer_assessment:
                            st.caption(producer_assessment)

                if filename_analysis:
                    st.markdown("**Filename analysis**")
                    for indicator in filename_analysis:
                        st.markdown(f"- {indicator}")

                if institutional_indicators:
                    st.markdown("**Institutional indicators found**")
                    for indicator in institutional_indicators:
                        st.markdown(f"- {indicator.title()}")

                if extracted_text:
                    with st.expander("Extracted text preview (first 200 characters)", expanded=False):
                        preview = metadata.get('extracted_text_full', extracted_text)[:200]
                        st.text(preview)
                        full_text = metadata.get('extracted_text_full', extracted_text)
                        if len(full_text) > 200:
                            st.caption(f"Full text available ({len(full_text)} characters).")
                            with st.expander("View full extracted text", expanded=False):
                                st.text(full_text)

        # --- Detector outputs & raw data ---
        with det_tab:
            st.markdown("### 📊 Detector outputs & raw data")

            if analysis['file_type'] == 'pdf' and analysis.get('all_results'):
                all_results = analysis['all_results']

                with st.expander("PDF structure", expanded=False):
                    if 'structure' in all_results:
                        structure = all_results['structure']
                        if structure.get('findings'):
                            st.json(structure['findings'])
                        st.caption(f"Risk score: {structure.get('risk_score', 0)}/100")
                    else:
                        st.info("Structure analysis not available.")

                with st.expander("Fonts", expanded=False):
                    if 'font' in all_results:
                        font = all_results['font']
                        if font.get('findings'):
                            st.json(font['findings'])
                        st.caption(f"Risk score: {font.get('risk_score', 0)}/100")
                    else:
                        st.info("Font analysis not available.")

                with st.expander("Text layer", expanded=False):
                    if 'text_layer' in all_results:
                        text_layer = all_results['text_layer']
                        if text_layer.get('findings'):
                            st.json(text_layer['findings'])
                        st.caption(f"Risk score: {text_layer.get('risk_score', 0)}/100")
                    else:
                        st.info("Text layer analysis not available.")

                with st.expander("Layout", expanded=False):
                    if 'layout' in all_results:
                        layout = all_results['layout']
                        if layout.get('findings'):
                            st.json(layout['findings'])
                        st.caption(f"Risk score: {layout.get('risk_score', 0)}/100")
                    else:
                        st.info("Layout analysis not available.")

                with st.expander("Signatures", expanded=False):
                    if 'signature' in all_results:
                        signature = all_results['signature']
                        if signature.get('findings'):
                            st.json(signature['findings'])
                        st.caption(f"Risk score: {signature.get('risk_score', 0)}/100")
                    else:
                        st.info("Signature analysis not available.")

                with st.expander("Embedded objects", expanded=False):
                    if 'embedded' in all_results:
                        embedded = all_results['embedded']
                        if embedded.get('findings'):
                            st.json(embedded['findings'])
                        st.caption(f"Risk score: {embedded.get('risk_score', 0)}/100")
                    else:
                        st.info("Embedded object analysis not available.")

            with st.expander("Noise variance analysis", expanded=False):
                st.write(analysis['noise']['findings'])

            with st.expander("Complete metadata extraction", expanded=False):
                st.json(analysis['metadata']['raw_data'])
    
    # Forensic Report Tab
    with main_tab3:
        st.markdown("## 📄 Comprehensive Forensic Report")
        st.markdown("")
        st.markdown("This report provides a comprehensive narrative summary of all forensic findings.")
        st.markdown("")
        
        # Retrieve Policy Result from Session State
        policy_result = st.session_state.get('current_policy_result')
        doc_type_label = st.session_state.get('current_doc_type_label')
        
        # Pass them to the report generators
        report = generate_forensic_report(analysis, policy_result, doc_type_label)
        formatted_report = format_report_for_display(analysis, policy_result, doc_type_label)
        
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
