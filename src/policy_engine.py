import re

try:
    import yaml
except ImportError:
    yaml = None  # Install PyYAML for policy config: pip install PyYAML

_DEFAULT_CONFIG = {
    'global_thresholds': {'high_risk_score': 70, 'medium_risk_score': 40},
    'document_types': {}
}


class PolicyEngine:
    """
    Evaluates forensic analysis results against defined business policies.
    """

    def __init__(self, config_path='config/policies.yaml'):
        if yaml is None:
            self.config = _DEFAULT_CONFIG
            return
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = _DEFAULT_CONFIG
        if not self.config:
            self.config = _DEFAULT_CONFIG

    def evaluate(self, analysis, doc_type_key):
        """
        Apply policy rules to analysis results.

        Args:
            analysis: The dictionary returned by analyze_file()
            doc_type_key: Key matching a section in policies.yaml (e.g., 'visa_refusal')

        Returns:
            dict: {
                'verdict': 'RED'|'AMBER'|'GREEN',
                'reason': str,
                'action': str,
                'critical_flags_hit': list
            }
        """
        # 1. Get Policy for this Document Type
        policies = self.config.get('document_types', {})
        policy = policies.get(doc_type_key, {})

        # Defaults if policy not found
        if not policy:
            return self._evaluate_generic(analysis)

        # 2. Check "Auto-Reject" Rules
        # These are instant fail conditions defined in YAML
        auto_reject_rules = policy.get('auto_reject_if', [])
        for rule in auto_reject_rules:
            reject_reason = self._check_auto_reject(analysis, rule)
            if reject_reason:
                return {
                    'verdict': 'RED',
                    'reason': f"Policy Violation: {reject_reason}",
                    'action': "Reject immediately. Do not accept this document.",
                    'critical_flags_hit': [reject_reason]
                }

        # 3. Check Critical Flags
        # These are specific flags that are unacceptable for this doc type
        critical_flags_config = policy.get('critical_flags', [])
        hit_critical_flags = []

        # Flatten all detected flags from analysis
        detected_flags = analysis['metadata'].get('flags', []) + \
                         analysis['noise'].get('flags', [])

        # Check metadata flags against critical config
        for flag in detected_flags:
            flag_lower = flag.lower()
            for critical in critical_flags_config:
                # Simple substring matching
                if critical.lower() in flag_lower:
                    hit_critical_flags.append(flag)
                # specific handling for "author_is_person" mapping
                elif critical == "author_is_person" and "author field" in flag_lower:
                    hit_critical_flags.append(flag)

        if hit_critical_flags:
            return {
                'verdict': 'RED',
                'reason': f"Critical Irregularities detected: {hit_critical_flags[0]}",
                'action': "Escalate to manager. High probability of forgery.",
                'critical_flags_hit': hit_critical_flags
            }

        # 4. Check Risk Scores (Weighted)
        # We can apply custom weights from config here if needed
        risk_score = analysis['metadata']['risk_score']
        trust_score = analysis['metadata']['trust_score']

        # Adjust score based on weights if defined
        weights = policy.get('weights', {})
        if 'ai_content' in weights and analysis.get('ai_content', {}).get('is_ai_generated'):
            # Artificial score boost for AI content if weighted heavily
            risk_score = min(100, risk_score * weights['ai_content'])

        thresholds = self.config.get('global_thresholds', {})
        high_risk = thresholds.get('high_risk_score', 70)
        medium_risk = thresholds.get('medium_risk_score', 40)

        if risk_score >= high_risk or trust_score < 40:
            return {
                'verdict': 'RED',
                'reason': f"High Risk Score ({risk_score}/100)",
                'action': "Reject or request original digital evidence.",
                'critical_flags_hit': []
            }
        elif risk_score >= medium_risk:
            return {
                'verdict': 'AMBER',
                'reason': "Moderate Anomalies Detected",
                'action': "Manual review required. Verify with issuing institution if possible.",
                'critical_flags_hit': []
            }
        else:
            return {
                'verdict': 'GREEN',
                'reason': "Passes Forensic Checks",
                'action': "Accept as evidence.",
                'critical_flags_hit': []
            }

    def _check_auto_reject(self, analysis, rule):
        """
        Parses simple string rules like 'producer_contains: aspose' or 'ai_confidence > 80'
        """
        rule = rule.lower()

        # Producer/Software Checks
        if "producer_contains" in rule or "aspose" in rule or "ilovepdf" in rule:
            target = rule.split(":")[-1].strip() if ":" in rule else rule
            raw = analysis['metadata'].get('raw_data', {})
            pdf_meta = raw.get('pdf_metadata', {})
            producer = str(raw.get('producer', '') or pdf_meta.get('producer', '')).lower()
            creator = str(raw.get('creator', '') or pdf_meta.get('creator', '')).lower()
            software = str(raw.get('software', '')).lower()
            if target in producer or target in creator or target in software:
                return f"Banned Software Detected: {target.title()}"

        # AI Confidence Checks
        if "ai_confidence" in rule:
            # Parse "ai_confidence > 80"
            match = re.search(r'ai_confidence\s*>\s*(\d+)', rule)
            if match:
                threshold = int(match.group(1))
                ai_result = analysis.get('ai_content', {})
                if ai_result and ai_result.get('confidence', 0) > threshold:
                    return f"AI Confidence > {threshold}%"

        return None

    def _evaluate_generic(self, analysis):
        """Fallback for unknown document types"""
        risk = analysis['metadata']['risk_score']
        if risk >= 70:
            return {'verdict': 'RED', 'reason': 'Generic High Risk', 'action': 'Reject', 'critical_flags_hit': []}
        elif risk >= 40:
            return {'verdict': 'AMBER', 'reason': 'Generic Moderate Risk', 'action': 'Review', 'critical_flags_hit': []}
        return {'verdict': 'GREEN', 'reason': 'Low Risk', 'action': 'Accept', 'critical_flags_hit': []}
