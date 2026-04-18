"""
Confidence Scorer Module

Combines all fraud detection indicators to calculate high-confidence scores.
"""


class ConfidenceScorer:
    """Calculates confidence scores based on multiple fraud indicators."""

    # --- Pixel / fusion calibration (real-world benchmarks) ---
    # Noise variance: Laplacian variance on greyscale render (same as PixelDetector)
    NOISE_VARIANCE_GREEN_MAX = 800
    NOISE_VARIANCE_AMBER_MAX = 2000
    # > NOISE_VARIANCE_AMBER_MAX => RED

    # ELA: JPEG Q60 mean absolute diff per pixel (RGB mean), 0–255 scale
    ELA_Q60_GREEN_MAX = 5.0
    ELA_Q60_AMBER_MAX = 12.0
    # > ELA_Q60_AMBER_MAX => RED

    # DCT: count of 8×8 blocks whose HF energy z-score > 1.5 vs block population
    DCT_BLOCKS_GREEN_MAX = 5000
    DCT_BLOCKS_AMBER_MAX = 8000
    # > DCT_BLOCKS_AMBER_MAX => RED

    PRODUCER_PIXEL_WHITELIST = (
        "gov.uk",
        "nhs",
        "adobe acrobat",
        "microsoft",
        "quartz pdfcontext",
        "itext",
        "fpdf",
    )
    
    # Strong fraud indicators (high confidence triggers)
    STRONG_INDICATORS = [
        'suspicious_software', 'timeline_anomaly', 'digital_manipulation',
        'invalid_signatures', 'post_signing_modifications', 'xref_anomalies',
        'no_text_layer', 'ocr_artifacts', 'font_embedding_issues',
        'ela_anomalies', 'smoothing_detected'
    ]
    
    # Weak indicators (common in legitimate documents, only contribute to moderate suspicion)
    WEAK_INDICATORS = [
        'incremental_updates', 'no_signatures', 'object_count',
        'version_inconsistency', 'alignment_issues', 'font_size_issues',
        'margin_issues'
    ]
    
    # Indicator categories and their weights
    INDICATOR_WEIGHTS = {
        'metadata': {
            'suspicious_software': 30,
            'timeline_anomaly': 25,
            'missing_metadata': 15,
            'institutional_indicators': -10  # Reduces suspicion
        },
        'structure': {
            'incremental_updates': 5,  # Reduced - too common
            'xref_anomalies': 25,
            'version_inconsistency': 10  # Reduced
        },
        'content': {
            'font_embedding_issues': 20,
            'system_fonts': 25,
            'font_inconsistency': 15,
            'no_text_layer': 30,
            'ocr_artifacts': 15,
            'alignment_issues': 10,  # Reduced
            'font_size_issues': 8,  # Reduced
            'margin_issues': 8  # Reduced
        },
        'pixel': {
            'smoothing_detected': 20,
            'ela_anomalies': 25
        },
        'signature': {
            'invalid_signatures': 40,
            'timestamp_anomalies': 25,
            'post_signing_modifications': 50
        }
    }

    @staticmethod
    def _producer_whitelist_applies(metadata_result: dict) -> bool:
        raw = (metadata_result or {}).get("raw_data") or {}
        pdf_meta = raw.get("pdf_metadata") or {}
        producer = (
            str(raw.get("producer") or pdf_meta.get("producer") or "")
        ).lower()
        if not producer.strip():
            return False
        return any(token in producer for token in ConfidenceScorer.PRODUCER_PIXEL_WHITELIST)

    @staticmethod
    def _institutional_indicators_present(metadata_result: dict) -> bool:
        raw = (metadata_result or {}).get("raw_data") or {}
        ind = raw.get("institutional_indicators")
        if isinstance(ind, list):
            return len(ind) > 0
        if isinstance(ind, str) and ind.strip():
            return True
        return False

    def _mitigate_missing_metadata_for_institutional_docs(self, metadata_result: dict) -> None:
        """
        Missing author/creator (or stripped metadata) is common on genuine government PDFs
        when institutional identity is visible in the document body.
        """
        if not metadata_result or not self._institutional_indicators_present(metadata_result):
            return

        raw = metadata_result.get("raw_data") or {}
        incomplete = bool(raw.get("metadata_incomplete"))

        flags = list(metadata_result.get("flags") or [])
        if not flags:
            return

        removable_substrings = (
            "missing author",
            "missing metadata",
            "metadata intentionally stripped",
            "metadata incomplete",
            "requires verification (institutional content detected)",
            "high suspicion (institutional content detected)",
            "high suspicion (no institutional indicators)",
        )

        new_flags: list[str] = []
        removed_any = False
        risk_adjust = 0
        for f in flags:
            fl = f.lower()
            should_remove = any(s in fl for s in removable_substrings) and (
                incomplete or "metadata intentionally stripped" in fl
            )
            if should_remove:
                removed_any = True
                if "metadata intentionally stripped" in fl:
                    risk_adjust += 15
                elif "missing author" in fl or "missing metadata" in fl:
                    risk_adjust += 20
                continue
            new_flags.append(f)

        if not removed_any:
            return

        note = (
            "Metadata absent or limited — consistent with official government document "
            "data protection or publishing practice; institutional indicators present in content."
        )
        if note not in new_flags:
            new_flags.append(note)

        metadata_result["flags"] = new_flags

        rs = int(metadata_result.get("risk_score", 0) or 0)
        ts = int(metadata_result.get("trust_score", 0) or 0)
        metadata_result["risk_score"] = max(0, min(100, rs - risk_adjust))
        metadata_result["trust_score"] = max(0, min(100, ts + 10))

    def _pixel_band_score(self, value: float, green_max: float, amber_max: float) -> int:
        """0 = GREEN band, 1 = AMBER, 2 = RED."""
        if value <= green_max:
            return 0
        if value <= amber_max:
            return 1
        return 2

    def calculate_confidence(self, all_results):
        """
        Calculate overall confidence score from all detector results.
        
        Args:
            all_results: dict containing results from all detectors:
                - metadata: MetadataDetector results
                - structure: PDFStructureDetector results
                - font: FontDetector results
                - text_layer: TextLayerDetector results
                - layout: LayoutDetector results
                - pixel: PixelDetector results
                - signature: SignatureDetector results
                - embedded: EmbeddedObjectDetector results
                
        Returns:
            dict: {
                'confidence_score': float (0-100),
                'confidence_level': str,
                'fraud_probability': float (0-100),
                'indicator_count': int,
                'indicator_breakdown': dict,
                'recommendation': str
            }
        """
        # Collect all flags and risk scores
        all_flags = []
        category_scores = {}
        indicator_count = 0
        
        # Metadata indicators
        if 'metadata' in all_results:
            metadata_result = all_results['metadata']
            self._mitigate_missing_metadata_for_institutional_docs(metadata_result)
            all_flags.extend(metadata_result.get('flags', []))
            category_scores['metadata'] = metadata_result.get('risk_score', 0)
            if metadata_result.get('flags'):
                indicator_count += len(metadata_result['flags'])
        
        # Structure indicators
        if 'structure' in all_results:
            structure_result = all_results['structure']
            all_flags.extend(structure_result.get('flags', []))
            category_scores['structure'] = structure_result.get('risk_score', 0)
            if structure_result.get('flags'):
                indicator_count += len(structure_result['flags'])
        
        # Font indicators
        if 'font' in all_results:
            font_result = all_results['font']
            all_flags.extend(font_result.get('flags', []))
            category_scores['font'] = font_result.get('risk_score', 0)
            if font_result.get('flags'):
                indicator_count += len(font_result['flags'])
        
        # Text layer indicators
        if 'text_layer' in all_results:
            text_result = all_results['text_layer']
            all_flags.extend(text_result.get('flags', []))
            category_scores['text_layer'] = text_result.get('risk_score', 0)
            if text_result.get('flags'):
                indicator_count += len(text_result['flags'])
        
        # Layout indicators
        if 'layout' in all_results:
            layout_result = all_results['layout']
            all_flags.extend(layout_result.get('flags', []))
            category_scores['layout'] = layout_result.get('risk_score', 0)
            if layout_result.get('flags'):
                indicator_count += len(layout_result['flags'])

        # Pixel / fusion calibration (noise variance, ELA Q60, DCT block outliers)
        noise_result = all_results.get("noise") or {}
        metadata_for_pixel = all_results.get("metadata") or {}
        whitelist_factor = (
            0.5 if self._producer_whitelist_applies(metadata_for_pixel) else 1.0
        )

        noise_var = noise_result.get("variance")
        ela_q60_mean = noise_result.get("ela_q60_mean")
        dct_blocks_high = noise_result.get("dct_blocks_z_gt_1_5")

        pixel_subscores = []
        if isinstance(noise_var, (int, float)):
            adj_var = float(noise_var) * whitelist_factor
            pixel_subscores.append(
                self._pixel_band_score(
                    adj_var,
                    self.NOISE_VARIANCE_GREEN_MAX,
                    self.NOISE_VARIANCE_AMBER_MAX,
                )
            )
        if isinstance(ela_q60_mean, (int, float)) and float(ela_q60_mean) > 0:
            adj_ela = float(ela_q60_mean) * whitelist_factor
            pixel_subscores.append(
                self._pixel_band_score(
                    adj_ela,
                    self.ELA_Q60_GREEN_MAX,
                    self.ELA_Q60_AMBER_MAX,
                )
            )
        if isinstance(dct_blocks_high, int):
            adj_dct = float(dct_blocks_high) * whitelist_factor
            pixel_subscores.append(
                self._pixel_band_score(
                    adj_dct,
                    self.DCT_BLOCKS_GREEN_MAX,
                    self.DCT_BLOCKS_AMBER_MAX,
                )
            )

        if pixel_subscores:
            worst = max(pixel_subscores)
            if worst == 0:
                category_scores["pixel"] = 10
            elif worst == 1:
                category_scores["pixel"] = 40
            else:
                category_scores["pixel"] = 75

        # Pixel indicators (flags from noise pipeline)
        if 'noise' in all_results:
            pixel_result = all_results.get('noise', {})
            all_flags.extend(pixel_result.get('flags', []))
            # Pixel risk is already factored into metadata trust score
        
        # Signature indicators
        if 'signature' in all_results:
            signature_result = all_results['signature']
            all_flags.extend(signature_result.get('flags', []))
            category_scores['signature'] = signature_result.get('risk_score', 0)
            if signature_result.get('flags'):
                indicator_count += len(signature_result['flags'])
        
        # Embedded object indicators
        if 'embedded' in all_results:
            embedded_result = all_results['embedded']
            all_flags.extend(embedded_result.get('flags', []))
            category_scores['embedded'] = embedded_result.get('risk_score', 0)
            if embedded_result.get('flags'):
                indicator_count += len(embedded_result['flags'])
        
        # Separate strong and weak indicators
        strong_flags = []
        weak_flags = []
        
        # Strong indicator keywords (actual flag text patterns)
        strong_keywords = [
            'digital manipulation', 'suspicious software', 'timeline anomaly',
            'temporal metadata', 'photoshop', 'gimp', 'canva', 'sejda', 'ilovepdf',
            'invalid signature', 'post signing', 'cross-reference', 'xref',
            'no text layer', 'ocr artifact', 'font embedding', 'smoothing detected',
            'ela anomaly', 'error level analysis'
        ]
        
        # Weak indicator keywords (common in legitimate documents)
        weak_keywords = [
            'incremental', 'object count', 'version inconsistency',
            'alignment', 'font size', 'margin', 'no digital signature'
        ]
        
        for flag in all_flags:
            flag_lower = flag.lower()
            is_strong = any(keyword in flag_lower for keyword in strong_keywords)
            is_weak = any(keyword in flag_lower for keyword in weak_keywords)
            
            if is_strong:
                strong_flags.append(flag)
            elif is_weak:
                weak_flags.append(flag)
            else:
                # Default: check if it's a known strong indicator pattern
                if any(x in flag_lower for x in ['manipulation', 'fraud', 'anomaly', 'suspicious']):
                    strong_flags.append(flag)
                else:
                    # Unknown flags default to weak (conservative approach)
                    weak_flags.append(flag)
        
        strong_count = len(strong_flags)
        weak_count = len(weak_flags)
        
        # Calculate fraud probability (0-100) - only from strong indicators
        total_risk = sum(category_scores.values())
        fraud_probability = min(total_risk, 100)
        
        # Calculate confidence based on STRONG indicators only
        # Weak indicators only contribute to moderate suspicion, never high confidence
        unique_categories = len([c for c in category_scores.values() if c > 0])
        
        # Confidence calculation based on STRONG indicators:
        # - 3+ strong indicators = 90%+ confidence (Definitive Fraud)
        # - 2 strong indicators = 70-89% confidence (High Suspicion)
        # - 1 strong indicator = 50-69% confidence (Moderate Suspicion)
        # - 0 strong indicators but weak indicators = 30-49% confidence (Low Suspicion)
        # - 0 indicators = <30% confidence (Legitimate)
        
        if strong_count >= 3:
            confidence_score = 90 + min(10, (strong_count - 3) * 2)
            confidence_level = 'Definitive Fraud'
            recommendation = 'HIGH CONFIDENCE: Document shows multiple strong fraud indicators. No verification needed.'
        elif strong_count >= 2:
            confidence_score = 70 + min(19, (strong_count - 2) * 5)
            confidence_level = 'High Suspicion'
            recommendation = 'HIGH SUSPICION: Multiple strong fraud indicators detected. Document likely fraudulent.'
        elif strong_count >= 1:
            confidence_score = 50 + min(19, strong_count * 10)
            confidence_level = 'Moderate Suspicion'
            recommendation = 'MODERATE SUSPICION: Strong fraud indicator detected. Recommend verification.'
        elif weak_count >= 2:
            # Only weak indicators - lower confidence
            confidence_score = 30 + min(19, weak_count * 5)
            confidence_level = 'Low Suspicion'
            recommendation = 'LOW SUSPICION: Minor anomalies detected. Document likely legitimate but verify if needed.'
        elif weak_count >= 1:
            confidence_score = 20 + min(9, weak_count * 5)
            confidence_level = 'Low Suspicion'
            recommendation = 'LOW SUSPICION: Minor anomalies detected. Document appears legitimate.'
        else:
            confidence_score = max(0, 100 - fraud_probability)
            confidence_level = 'Low Suspicion'
            recommendation = 'LOW SUSPICION: Minimal fraud indicators. Document appears legitimate.'
        
        # Ensure confidence doesn't exceed 100
        confidence_score = min(confidence_score, 100)
        
        return {
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'fraud_probability': fraud_probability,
            'indicator_count': indicator_count,
            'unique_categories': unique_categories,
            'category_scores': category_scores,
            'all_flags': all_flags,
            'recommendation': recommendation,
            'indicator_breakdown': {
                'metadata_indicators': len([f for f in all_flags if any(x in f.lower() for x in ['metadata', 'software', 'timeline', 'author', 'creator'])]),
                'structure_indicators': len([f for f in all_flags if any(x in f.lower() for x in ['incremental', 'xref', 'version', 'structure'])]),
                'content_indicators': len([f for f in all_flags if any(x in f.lower() for x in ['font', 'text', 'layout', 'alignment', 'margin'])]),
                'pixel_indicators': len([f for f in all_flags if any(x in f.lower() for x in ['smoothing', 'noise', 'ela'])]),
                'signature_indicators': len([f for f in all_flags if 'signature' in f.lower()])
            }
        }
