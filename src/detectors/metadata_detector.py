"""
Metadata Detector Module

Extracts and scores EXIF/XMP data from documents.
"""

import fitz  # PyMuPDF
from PIL import Image
from PIL.ExifTags import TAGS
from io import BytesIO
from datetime import datetime


class MetadataDetector:
    """Detects fraud indicators through metadata analysis."""
    
    # Suspicious editing software
    SUSPICIOUS_SOFTWARE = [
        'photoshop', 'gimp', 'canva', 'sejda', 'ilovepdf',
        'adobe photoshop', 'gimp', 'paint.net', 'photopea'
    ]
    
    # High trust banking/document software
    TRUSTED_SOFTWARE = [
        'crystal reports', 'adobe livecycle', 'adobe acrobat',
        'microsoft word', 'microsoft excel', 'adobe indesign'
    ]
    
    # Institutional indicator patterns
    INSTITUTIONAL_PATTERNS = [
        # UK Universities
        'university', 'college', '.ac.uk', 'department', 'faculty',
        'admissions', 'registry', 'student services',
        # Home Office
        'home office', 'ukvi', 'uk visa', 'immigration', 'visa',
        'entry clearance', 'confirmation of acceptance',
        # Email domains
        '@ac.uk', '@gov.uk',
        # Document types
        'cas', 'confirmation of acceptance for studies'
    ]
    
    def analyze(self, file_stream, file_type, filename=''):
        """
        Analyze metadata for fraud indicators.
        
        Args:
            file_stream: File-like object (BytesIO or file handle)
            file_type: str, either 'pdf' or 'image'
            filename: Optional filename for additional analysis
            
        Returns:
            dict: {
                'risk_score': int (0-100, where 100 is highest risk),
                'flags': list of str,
                'raw_data': dict with extracted metadata,
                'trust_score': int (0-100, where 100 is highest trust)
            }
        """
        flags = []
        raw_data = {}
        risk_score = 0
        trust_score = 85  # Default high trust (assumes no manipulation until proven otherwise)
        
        try:
            if file_type.lower() == 'pdf':
                result = self._analyze_pdf(file_stream, filename)
            elif file_type.lower() in ['image', 'jpg', 'jpeg', 'png', 'tiff', 'tif']:
                result = self._analyze_image(file_stream)
            else:
                return {
                    'risk_score': 0,
                    'flags': ['Unsupported File Format'],
                    'raw_data': {},
                    'trust_score': 0
                }
            
            flags = result['flags']
            raw_data = result['raw_data']
            risk_score = result['risk_score']
            trust_score = result['trust_score']
            
        except Exception as e:
            flags.append(f'Metadata Extraction Error: {str(e)}')
            raw_data['error'] = str(e)
        
        return {
            'risk_score': risk_score,
            'flags': flags,
            'raw_data': raw_data,
            'trust_score': trust_score
        }
    
    def _extract_pdf_text(self, pdf_doc, max_pages=3):
        """
        Extract text from first few pages of PDF.
        
        Args:
            pdf_doc: PyMuPDF document object
            max_pages: Maximum number of pages to extract (default 3)
            
        Returns:
            str: Combined text from pages
        """
        text_parts = []
        page_count = min(len(pdf_doc), max_pages)
        
        for page_num in range(page_count):
            try:
                page = pdf_doc[page_num]
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception:
                continue
        
        return '\n\n'.join(text_parts)
    
    def _find_institutional_indicators(self, text, filename=''):
        """
        Search for institutional clues in text and filename.
        
        Args:
            text: Document text to search
            filename: Filename to search
            
        Returns:
            list: Found institutional indicators
        """
        indicators = []
        search_text = (text + ' ' + filename).lower()
        
        for pattern in self.INSTITUTIONAL_PATTERNS:
            if pattern.lower() in search_text:
                indicators.append(pattern)
        
        # Check for UK phone numbers (basic pattern)
        import re
        uk_phone_pattern = r'\+44\s?\d{1,4}\s?\d{3,4}\s?\d{3,4}'
        if re.search(uk_phone_pattern, text):
            indicators.append('UK Phone Number')
        
        return list(set(indicators))  # Remove duplicates
    
    def _check_metadata_completeness(self, metadata):
        """
        Check if critical metadata fields are missing.
        
        Args:
            metadata: PDF metadata dictionary
            
        Returns:
            dict: {
                'is_complete': bool,
                'missing_fields': list,
                'has_author': bool,
                'has_creator': bool,
                'has_producer': bool
            }
        """
        author = (metadata.get('author') or '').strip()
        creator = (metadata.get('creator') or '').strip()
        producer = (metadata.get('producer') or '').strip()
        
        has_author = bool(author)
        has_creator = bool(creator)
        has_producer = bool(producer)
        
        missing_fields = []
        if not has_author:
            missing_fields.append('author')
        if not has_creator:
            missing_fields.append('creator')
        
        is_complete = has_author and has_creator
        
        return {
            'is_complete': is_complete,
            'missing_fields': missing_fields,
            'has_author': has_author,
            'has_creator': has_creator,
            'has_producer': has_producer
        }
    
    def _analyze_pdf(self, file_stream, filename=''):
        """Analyze PDF metadata."""
        flags = []
        raw_data = {}
        risk_score = 0
        trust_score = 85  # Default high trust (assumes no manipulation until proven otherwise)
        
        # Reset stream position
        file_stream.seek(0)
        
        # Open PDF with PyMuPDF
        pdf_doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        metadata = pdf_doc.metadata
        
        raw_data['pdf_metadata'] = metadata
        
        # Store metadata fields (convert empty strings to None)
        raw_data['author'] = (metadata.get('author') or '').strip() or None
        raw_data['creator'] = (metadata.get('creator') or '').strip() or None
        raw_data['producer'] = (metadata.get('producer') or '').strip() or None
        
        # Check metadata completeness
        completeness = self._check_metadata_completeness(metadata)
        raw_data['metadata_completeness'] = completeness
        
        # Extract text from PDF for alternative analysis (do this BEFORE metadata check to inform risk scoring)
        institutional_indicators = []
        extracted_text = None
        try:
            extracted_text = self._extract_pdf_text(pdf_doc, max_pages=3)
            if extracted_text:
                # Store first 500 chars for storage efficiency
                raw_data['extracted_text'] = extracted_text[:500]
                raw_data['extracted_text_full'] = extracted_text  # Full text available
                
                # Find institutional indicators (include filename in search)
                indicators = self._find_institutional_indicators(extracted_text, filename)
                if indicators:
                    institutional_indicators = indicators
                    raw_data['institutional_indicators'] = indicators
                    flags.append('Institutional Indicators Found in Document Content')
        except Exception as e:
            # Text extraction failed, continue without it
            raw_data['text_extraction_error'] = str(e)
        
        # Track sources of risk for trust score calculation
        has_actual_manipulation = False  # Photoshop, timeline anomalies, etc.
        has_missing_metadata_with_indicators = False
        has_missing_metadata_without_indicators = False
        
        # Enhanced metadata completeness check with institutional indicator consideration
        if not completeness['is_complete']:
            raw_data['metadata_incomplete'] = True
            
            # Check if Producer is a generic system tool
            producer_lower = (raw_data['producer'] or '').lower()
            is_generic_producer = any(term in producer_lower for term in [
                'quartz', 'pdfcontext', 'print', 'system', 'default',
                'microsoft print to pdf', 'save as pdf'
            ])
            
            # Adjust risk score based on institutional indicators
            if institutional_indicators:
                # Institutional content found - likely legitimate but re-saved
                risk_score += 15  # Reduced from 25
                has_missing_metadata_with_indicators = True
                if is_generic_producer:
                    flags.append('Missing Metadata - Document Likely Re-Saved/Printed (Institutional Content Detected)')
                    raw_data['assessment'] = 'Likely legitimate document that was re-saved/printed. Institutional indicators present but metadata lost during re-processing.'
                else:
                    flags.append('Missing Metadata - Requires Verification (Institutional Content Detected)')
                    raw_data['assessment'] = 'Institutional content detected but metadata incomplete. Verify authenticity with issuing institution.'
            else:
                # No institutional indicators - higher suspicion
                risk_score += 25  # Full penalty
                has_missing_metadata_without_indicators = True
                if is_generic_producer:
                    flags.append('Missing Author/Creator Metadata - High Suspicion (No Institutional Indicators)')
                    raw_data['assessment'] = 'Missing metadata and no institutional indicators found. Document may be fraudulent or improperly created.'
                else:
                    flags.append('Missing Author/Creator Metadata - High Suspicion')
                    raw_data['assessment'] = 'Missing critical metadata. Verify document authenticity.'
        
        # Filename analysis
        if filename:
            raw_data['filename'] = filename
            # Check filename for document type patterns
            filename_lower = filename.lower()
            filename_indicators = []
            if 'cas' in filename_lower:
                filename_indicators.append('CAS Document')
            if 'visa' in filename_lower:
                filename_indicators.append('Visa Document')
            if any(term in filename_lower for term in ['university', 'college', 'ac.uk']):
                filename_indicators.append('University Document')
            if filename_indicators:
                raw_data['filename_analysis'] = filename_indicators
        
        # Producer field analysis
        if raw_data['producer']:
            producer_lower = raw_data['producer'].lower()
            is_generic_producer = any(term in producer_lower for term in [
                'quartz', 'pdfcontext', 'print', 'system', 'default',
                'microsoft print to pdf', 'save as pdf'
            ])
            
            if not completeness['has_author'] and not completeness['has_creator']:
                if is_generic_producer:
                    raw_data['producer_note'] = f"Producer: {raw_data['producer']} (Generic system tool - document likely printed/saved)"
                    if institutional_indicators:
                        raw_data['producer_assessment'] = 'Document appears to be legitimate institutional content that was printed/saved, causing metadata loss.'
                else:
                    raw_data['producer_note'] = f"Producer Available: {raw_data['producer']}"
            else:
                if is_generic_producer:
                    raw_data['producer_note'] = f"Producer: {raw_data['producer']} (Generic system tool)"
        
        # Check creation date vs modification date
        creation_date = metadata.get('creationDate', '')
        mod_date = metadata.get('modDate', '')
        
        if creation_date and mod_date:
            try:
                # Parse dates (format: D:YYYYMMDDHHmmSSOHH'mm)
                if creation_date.startswith('D:'):
                    creation_date = creation_date[2:]
                if mod_date.startswith('D:'):
                    mod_date = mod_date[2:]
                
                # Extract date parts (YYYYMMDD)
                if len(creation_date) >= 8 and len(mod_date) >= 8:
                    creation_str = creation_date[:8]
                    mod_str = mod_date[:8]
                    
                    creation_dt = datetime.strptime(creation_str, '%Y%m%d')
                    mod_dt = datetime.strptime(mod_str, '%Y%m%d')
                    
                    if mod_dt < creation_dt:
                        flags.append('Temporal Metadata Inconsistency')
                        risk_score += 30
                        has_actual_manipulation = True  # Timeline anomaly is actual manipulation
                        raw_data['timeline_anomaly'] = {
                            'creation': creation_dt.isoformat(),
                            'modification': mod_dt.isoformat()
                        }
            except (ValueError, IndexError) as e:
                # Date parsing failed, skip this check
                pass
        
        # Check Producer and Creator fields (already stored above, but need for analysis)
        producer = (metadata.get('producer') or '').lower()
        creator = (metadata.get('creator') or '').lower()
        
        # Check for suspicious software
        found_suspicious = False
        detected_software = None
        for suspicious in self.SUSPICIOUS_SOFTWARE:
            if suspicious in producer or suspicious in creator:
                # Format professional message
                if 'photoshop' in suspicious:
                    flags.append('Digital Manipulation Software Detected (Adobe Photoshop)')
                elif 'gimp' in suspicious:
                    flags.append('Digital Manipulation Software Detected (GIMP)')
                elif 'canva' in suspicious:
                    flags.append('Digital Manipulation Software Detected (Canva)')
                else:
                    flags.append(f'Digital Manipulation Software Detected ({suspicious.title()})')
                risk_score += 40
                found_suspicious = True
                has_actual_manipulation = True  # Suspicious software is actual manipulation
                detected_software = suspicious
                raw_data['suspicious_software'] = suspicious
                break
        
        # Check for trusted software
        found_trusted = False
        for trusted in self.TRUSTED_SOFTWARE:
            if trusted in producer or trusted in creator:
                found_trusted = True
                raw_data['trusted_software'] = trusted
                break
        
        # Determine trust score based on findings
        # Priority: Actual manipulation > Missing metadata with indicators > Missing metadata without indicators
        
        if has_actual_manipulation:
            # Actual manipulation detected (Photoshop, timeline anomalies, etc.)
            trust_score = 15  # Low trust - manipulation detected
        elif found_trusted:
            # Trusted software found (Crystal Reports, Adobe LiveCycle, etc.)
            trust_score = 90  # High trust - trusted software and no manipulation
        elif has_missing_metadata_with_indicators:
            # Missing metadata BUT institutional indicators present
            # Document likely legitimate but re-saved/printed
            trust_score = 65  # Moderate trust - verify but likely legitimate
            raw_data['trust_reason'] = 'Missing metadata mitigated by institutional indicators'
        elif has_missing_metadata_without_indicators:
            # Missing metadata AND no institutional indicators
            trust_score = 45  # Lower trust - more suspicious
            raw_data['trust_reason'] = 'Missing metadata with no institutional indicators'
        else:
            # No issues found
            trust_score = 85  # High trust - no manipulation indicators found
        
        pdf_doc.close()
        
        return {
            'flags': flags,
            'raw_data': raw_data,
            'risk_score': min(risk_score, 100),
            'trust_score': trust_score
        }
    
    def _analyze_image(self, file_stream):
        """Analyze image EXIF metadata."""
        flags = []
        raw_data = {}
        risk_score = 0
        trust_score = 85  # Default high trust (assumes no manipulation until proven otherwise)
        
        # Reset stream position
        file_stream.seek(0)
        
        # Open image with PIL
        image = Image.open(file_stream)
        
        # Extract EXIF data
        exif_data = image.getexif()
        exif_dict = {}
        
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_dict[tag] = value
                raw_data[tag] = str(value)
        
        raw_data['exif_data'] = exif_dict
        
        # Check Software tag
        software = exif_dict.get('Software', '').lower()
        raw_data['software'] = exif_dict.get('Software', '')
        
        # Check for suspicious software
        found_suspicious = False
        detected_software = None
        for suspicious in self.SUSPICIOUS_SOFTWARE:
            if suspicious in software:
                # Format professional message
                if 'photoshop' in suspicious:
                    flags.append('Digital Manipulation Software Detected (Adobe Photoshop)')
                elif 'gimp' in suspicious:
                    flags.append('Digital Manipulation Software Detected (GIMP)')
                elif 'canva' in suspicious:
                    flags.append('Digital Manipulation Software Detected (Canva)')
                else:
                    flags.append(f'Digital Manipulation Software Detected ({suspicious.title()})')
                risk_score += 40
                found_suspicious = True
                detected_software = suspicious
                raw_data['suspicious_software'] = suspicious
                break
        
        # Check for trusted software
        found_trusted = False
        for trusted in self.TRUSTED_SOFTWARE:
            if trusted in software:
                found_trusted = True
                raw_data['trusted_software'] = trusted
                break
        
        # Determine trust score based on findings
        # Priority: Manipulation detection overrides everything
        has_manipulation = found_suspicious or (risk_score > 0)  # Any suspicious software
        
        if has_manipulation:
            trust_score = 15  # Low trust - manipulation detected
        elif found_trusted:
            trust_score = 90  # High trust - trusted software and no manipulation
        else:
            trust_score = 85  # High trust - no manipulation indicators found
        
        image.close()
        
        return {
            'flags': flags,
            'raw_data': raw_data,
            'risk_score': min(risk_score, 100),
            'trust_score': trust_score
        }
