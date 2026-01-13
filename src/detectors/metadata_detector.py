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
    
    def analyze(self, file_stream, file_type):
        """
        Analyze metadata for fraud indicators.
        
        Args:
            file_stream: File-like object (BytesIO or file handle)
            file_type: str, either 'pdf' or 'image'
            
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
                result = self._analyze_pdf(file_stream)
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
    
    def _analyze_pdf(self, file_stream):
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
                        raw_data['timeline_anomaly'] = {
                            'creation': creation_dt.isoformat(),
                            'modification': mod_dt.isoformat()
                        }
            except (ValueError, IndexError) as e:
                # Date parsing failed, skip this check
                pass
        
        # Check Producer and Creator fields
        producer = metadata.get('producer', '').lower()
        creator = metadata.get('creator', '').lower()
        
        raw_data['producer'] = metadata.get('producer', '')
        raw_data['creator'] = metadata.get('creator', '')
        
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
        # Priority: Manipulation detection overrides everything
        has_manipulation = found_suspicious or (risk_score > 0)  # Any suspicious software or timeline anomaly
        
        if has_manipulation:
            trust_score = 15  # Low trust - manipulation detected
        elif found_trusted:
            trust_score = 90  # High trust - trusted software and no manipulation
        else:
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
