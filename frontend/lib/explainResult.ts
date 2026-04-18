/**
 * Rule-based plain-English explanation of a forensic scan — no external APIs.
 */

export type PolicyVerdict = "RED" | "AMBER" | "GREEN";

export interface ExplainResultInput {
  verdict: PolicyVerdict;
  forgeryScore: number;
  trustScore: number;
  redFlags: string[];
  documentClassKey: string;
  documentClassLabel: string;
  detectedLanguage: string;
}

type FlagKind =
  | "DATE_DISCREPANCY"
  | "CONSUMER_SOFTWARE"
  | "TIMEZONE_MISMATCH"
  | "BLANK_SIGNATURE"
  | "INTERNAL_INCONSISTENCY"
  | "TEXT_POSITIONING"
  | "IMAGE_COMPRESSION"
  | "WPS_CREATOR"
  | "LANGUAGE_ISSUER"
  | "AI_CONTENT"
  | "METADATA_ANOMALY"
  | "GENERIC_HIGH";

/** Higher = more severe for ordering (top 3). */
const KIND_RANK: Record<FlagKind, number> = {
  DATE_DISCREPANCY: 100,
  INTERNAL_INCONSISTENCY: 96,
  CONSUMER_SOFTWARE: 94,
  WPS_CREATOR: 93,
  TIMEZONE_MISMATCH: 91,
  LANGUAGE_ISSUER: 90,
  IMAGE_COMPRESSION: 88,
  BLANK_SIGNATURE: 85,
  AI_CONTENT: 82,
  TEXT_POSITIONING: 78,
  METADATA_ANOMALY: 72,
  GENERIC_HIGH: 60,
};

function norm(s: string): string {
  return s.toLowerCase();
}

function classifyFlag(flag: string): FlagKind {
  const f = norm(flag);

  if (f.includes("document date does not match pdf creation") || (f.includes("pdf creation") && f.includes("calendar days"))) {
    return "DATE_DISCREPANCY";
  }
  if (f.includes("wps") && (f.includes("creator") || f.includes("producer") || f.includes("software"))) {
    return "WPS_CREATOR";
  }
  if (f.includes("personal software") || f.includes("consumer") || f.includes("institutional document systems")) {
    return "CONSUMER_SOFTWARE";
  }
  if (f.includes("timezone inconsistent") || f.includes("timezone")) {
    return "TIMEZONE_MISMATCH";
  }
  if (f.includes("unfilled") || f.includes("placeholder field")) {
    return "BLANK_SIGNATURE";
  }
  if (f.includes("signature") && (f.includes("blank") || f.includes("empty") || f.includes("template"))) {
    return "BLANK_SIGNATURE";
  }
  if (f.includes("internal inconsistency") || f.includes("appears differently across")) {
    return "INTERNAL_INCONSISTENCY";
  }
  if (f.includes("language is inconsistent") || f.includes("language vs issuer")) {
    return "LANGUAGE_ISSUER";
  }
  if (f.includes("ai-generated") || f.includes("ai content")) {
    return "AI_CONTENT";
  }
  if (
    f.includes("smoothing") ||
    f.includes("noise") ||
    f.includes("ela") ||
    f.includes("manipulation") ||
    f.includes("suspicious region") ||
    f.includes("anomaly cluster") ||
    f.includes("dct") ||
    f.includes("compression")
  ) {
    return "IMAGE_COMPRESSION";
  }
  if (
    f.includes("layout") ||
    f.includes("alignment") ||
    f.includes("margin") ||
    f.includes("font inconsistency") ||
    f.includes("text layer") ||
    f.includes("embedding")
  ) {
    return "TEXT_POSITIONING";
  }
  if (
    f.includes("metadata") ||
    f.includes("author") ||
    f.includes("creator") ||
    f.includes("timeline") ||
    f.includes("modified") ||
    f.includes("correlation")
  ) {
    return "METADATA_ANOMALY";
  }

  return "GENERIC_HIGH";
}

function extractCalendarDays(flag: string): number | null {
  const m = flag.match(/(\d+)\s+calendar days/i);
  if (m) return parseInt(m[1], 10);
  return null;
}

function inferRegionPhrase(flag: string): string {
  const f = norm(flag);
  if (f.includes("gov.uk") || f.includes("nhs") || f.includes("united kingdom") || f.includes("britain")) {
    return "the United Kingdom";
  }
  if (f.includes("china") || f.includes("chinese") || f.includes(".cn")) {
    return "China";
  }
  if (f.includes("united states") || f.includes(" u.s.")) {
    return "the United States";
  }
  return "the stated issuing region";
}

function sentenceForKind(kind: FlagKind, flag: string): string {
  const days = extractCalendarDays(flag);
  const region = inferRegionPhrase(flag);

  switch (kind) {
    case "DATE_DISCREPANCY":
      if (days != null && !Number.isNaN(days)) {
        return `The date written on the document does not match when the file was actually created on a computer — the timeline differs by about ${days} day${days === 1 ? "" : "s"}, which is unusual for a genuine institutional document.`;
      }
      return "The date written on the document does not match when the file was actually created on a computer — it was created on a different timeline than the stated date, which is unusual for a genuine institutional document.";
    case "CONSUMER_SOFTWARE":
      return "The document was created using personal software (such as WPS Office or Microsoft Word) rather than the official systems typically used by hospitals, universities, or government bodies.";
    case "WPS_CREATOR":
      return "The document was produced using WPS Office, a consumer word processor that is sometimes used to create inauthentic documents.";
    case "TIMEZONE_MISMATCH":
      return `The document appears to be associated with ${region}, but the file was created in a timezone that does not match that location — this can be a sign the document did not originate where it claims.`;
    case "BLANK_SIGNATURE":
      return "The signature field on this document appears to be empty or contains a placeholder, suggesting the document may be a template that was not properly completed.";
    case "INTERNAL_INCONSISTENCY":
      return "Information within the document is inconsistent — the same detail appears differently in different parts of the document.";
    case "TEXT_POSITIONING":
      return "The layout and positioning of text (or fonts / structure) on the document does not fully match what we would expect from a cleanly produced official file.";
    case "IMAGE_COMPRESSION":
      return "Parts of the document image show signs of digital editing, re-saving, or unusual noise/compression patterns, which can indicate tampering.";
    case "LANGUAGE_ISSUER":
      return "The language of the document text does not align well with the country or institution the document claims to represent.";
    case "AI_CONTENT":
      return "Automated checks suggest some wording patterns are consistent with AI-generated text, which may warrant a closer read.";
    case "METADATA_ANOMALY":
      return "Metadata or file history for this document shows anomalies (timestamps, authorship, or correlation signals) that deserve verification.";
    case "GENERIC_HIGH":
    default:
      return flag.length > 280 ? `${flag.slice(0, 277)}…` : flag;
  }
}

function part1(input: ExplainResultInput): string {
  const { verdict, forgeryScore, trustScore, redFlags } = input;
  const n = redFlags.length;

  let lead: string;
  if (verdict === "RED") {
    lead =
      "This document has been flagged as potentially fraudulent and should not be accepted without further verification.";
  } else if (verdict === "AMBER" || (verdict === "GREEN" && n > 0)) {
    lead =
      "This document passed our automated checks but has some minor anomalies worth noting.";
  } else {
    lead = "This document passed our automated checks and shows no significant signs of tampering.";
  }

  const fs = Math.round(Math.max(0, Math.min(100, forgeryScore)));
  const ts = Math.round(Math.max(0, Math.min(100, trustScore)));
  return `${lead} The forgery risk score is ${fs} out of 100, and the trust score is ${ts} out of 100.`;
}

function part2(input: ExplainResultInput): string {
  if (!input.redFlags.length) {
    return "";
  }

  const scored = input.redFlags.map((flag) => ({
    flag,
    kind: classifyFlag(flag),
    rank: KIND_RANK[classifyFlag(flag)],
  }));

  scored.sort((a, b) => b.rank - a.rank);

  const seen = new Set<FlagKind>();
  const sentences: string[] = [];
  for (const item of scored) {
    if (seen.has(item.kind)) continue;
    seen.add(item.kind);
    sentences.push(sentenceForKind(item.kind, item.flag));
    if (sentences.length >= 3) break;
  }

  if (!sentences.length) {
    return "";
  }
  return sentences.map((s) => `• ${s}`).join("\n");
}

function part3(input: ExplainResultInput): string {
  const { verdict, redFlags } = input;
  const n = redFlags.length;

  if (verdict === "RED") {
    return "Do not accept this document. Contact the student to request an original certified copy directly from the issuing institution. If fraud is suspected, escalate to your compliance team.";
  }

  if (verdict === "GREEN" && n === 0) {
    return "No further action is required. You may proceed with this document.";
  }

  if (n >= 1 && n <= 2) {
    return "We recommend requesting the original physical document or contacting the issuing institution directly to verify authenticity before proceeding.";
  }

  if (n >= 3) {
    return "We recommend requesting the original physical document or contacting the issuing institution directly to verify authenticity, and reviewing each flagged issue before proceeding.";
  }

  if (verdict === "AMBER") {
    return "We recommend requesting the original physical document or contacting the issuing institution directly to verify authenticity before proceeding.";
  }

  return "No further action is required. You may proceed with this document.";
}

/**
 * Build a full plain-English explanation (rule-based only).
 */
export function explainResult(input: ExplainResultInput): string {
  const p1 = part1(input);
  const p2 = part2(input);
  const p3 = part3(input);

  const docLine = `Document type: ${input.documentClassLabel}. Detected language (where available): ${input.detectedLanguage}.`;

  const blocks = [docLine, "", p1];
  if (p2) {
    blocks.push("", "Notable findings (top issues):", p2);
  }
  blocks.push("", "Recommended action:", p3);

  return blocks.join("\n");
}
