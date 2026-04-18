export interface PdfExtractedFields {
  dates: string[];
  names: string[];
  referenceNumbers: string[];
  issuingInstitution: string | null;
  signaturePresent: boolean;
}

export interface PdfExtractionResult {
  rawText: string;
  translatedText: string;
  detectedLanguage: string;
  confidence: number;
  extractedFields: PdfExtractedFields;
}
