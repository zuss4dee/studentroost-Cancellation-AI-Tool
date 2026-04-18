/**
 * Universal document content checks — generic heuristics only (no single-document hardcoding).
 * Runs client-side on extracted text + File DNA rows from any PDF.
 */

export type ContentFlagSeverity = "high" | "medium";

export type ContentFlagType =
  | "date_discrepancy"
  | "consumer_software"
  | "timezone_issuer"
  | "placeholder_field"
  | "internal_inconsistency"
  | "language_issuer";

export interface ContentFlag {
  type: ContentFlagType;
  severity: ContentFlagSeverity;
  label: string;
  detail: string;
}

export interface ContentAnalysisSummary {
  writtenDatesDisplay: string[];
  primaryLanguage: string;
  creatorSoftware: string;
  producerSoftware: string;
  pdfCreationRaw: string;
  /** IANA-style offset label e.g. UTC+01:00, or — */
  timezoneLabel: string;
  institutionHints: string[];
}

export interface UniversalCheckInput {
  extractedText: string;
  fileDnaRows: ReadonlyArray<{ property: string; value: string }>;
}

const MONTHS: Record<string, number> = {
  january: 1,
  february: 2,
  march: 3,
  april: 4,
  may: 5,
  june: 6,
  july: 7,
  august: 8,
  september: 9,
  october: 10,
  november: 11,
  december: 12,
};

const CONSUMER_CREATOR_RE =
  /wps|microsoft\s*word|ms\s*word|libreoffice|google\s*docs|docs\.google|pages|openoffice|neooffice|iwork/i;

/** Broad institutional cues — domains and org vocabulary, not tied to one issuer. */
const INSTITUTIONAL_CUES =
  /\.(edu|ac\.[a-z]{2}|gov\.[a-z]{2}|gouv\.|go\.jp|go\.kr|gc\.ca)\b|\b(university|college|polytechnic|hospital|clinic|ministry|department|authority|council|agency|national\s+health|public\s+health)\b|[\u4e00-\u9fff\u3400-\u4dbf](?:\s*[\u4e00-\u9fff\u3400-\u4dbf]){0,20}(?:大学|学院|政府|部|局|医院)/i;

const ENGLISH_SPEAKING_COUNTRY_CUES =
  /\b(united\s+kingdom|uk\b|great\s+britain|britain|england|scotland|wales|northern\s+ireland|ireland|united\s+states|u\.s\.a\.?|\busa\b|canada|australia|new\s+zealand)\b|\.(gov\.uk|ac\.uk|edu\.au|gc\.ca)\b/i;

const CHINA_REGION_CUES =
  /\b(china|people'?s\s+republic|prc|beijing|shanghai|guangzhou|shenzhen|hong\s+kong\s+sar|macau|taiwan|r\.o\.c\.)\b|\.cn\b|中国|中華|香港|澳門|台灣/;

const PLACEHOLDER_AFTER_LABEL_RE = new RegExp(
  [
    "(Signature|Name\\s*\\(print\\)?|Author|Prepared\\s*by|Approved\\s*by|Reviewer|Signatory)\\s*[:：]\\s*",
    "(",
    [
      "N\\/A",
      "TBC",
      "TBD",
      "\\[\\s*Name\\s*\\]",
      "XXXXX?",
      "x{3,}",
      "\\.{3,}",
      "—{2,}",
      "-{3,}",
      "\\bData\\b",
      "\\bAdmin\\b",
      "\\bUser\\b",
      "\\bOwner\\b",
      "\\bLorem\\s+ipsum\\b",
    ].join("|"),
    ")",
  ].join(""),
  "gi",
);

const BLANK_AFTER_LABEL_RE =
  /(Signature|Name\s*\(print\)?|Author|Date\s*signed|Signatory)\s*[:：]\s*($|\n|\r|<\/?[^>]+>)/gim;

const LABELED_VALUE_RE =
  /(Reference|Ref\.?|Case\s*(?:ID|No\.?)?|Account\s*(?:No\.?)?|File\s*(?:No\.?)?|ID|Tracking|Confirmation)\s*[:#]?\s*([^\n\r]+)/gi;

function dnaValue(rows: UniversalCheckInput["fileDnaRows"], prop: string): string {
  const row = rows.find((r) => r.property.trim().toLowerCase() === prop.toLowerCase());
  if (!row) return "";
  const v = row.value.trim();
  return v === "—" ? "" : v;
}

/** PDF metadata date: D:YYYYMMDDHHmmSSOHH'mm' or variants */
function parsePdfMetadataDate(raw: string): { utc: Date; offsetMinutes: number | null } | null {
  const s = raw.trim();
  if (!s) return null;
  const m = s.match(
    /D:(\d{4})(\d{2})(\d{2})(?:(\d{2})(?:(\d{2})(?:(\d{2}))?)?)?\s*(Z|[+\-]\d{2}'?\d{2}'?)?/i,
  );
  if (!m) {
    const iso = Date.parse(s);
    if (!Number.isNaN(iso)) return { utc: new Date(iso), offsetMinutes: null };
    return null;
  }
  const y = +m[1];
  const mon = +m[2];
  const d = +m[3];
  const hh = m[4] != null ? +m[4] : 0;
  const mm = m[5] != null ? +m[5] : 0;
  const ss = m[6] != null ? +m[6] : 0;
  let offsetMinutes: number | null = null;
  const tz = m[7];
  if (tz && /^[+\-]/.test(tz)) {
    const tzm = tz.replace(/'/g, "").match(/^([+\-])(\d{2})(\d{2})$/);
    if (tzm) {
      const sign = tzm[1] === "-" ? -1 : 1;
      offsetMinutes = sign * (parseInt(tzm[2], 10) * 60 + parseInt(tzm[3], 10));
    }
  } else if (tz && /^z$/i.test(tz)) {
    offsetMinutes = 0;
  }
  const asUtc = Date.UTC(y, mon - 1, d, hh, mm, ss);
  return { utc: new Date(asUtc), offsetMinutes };
}

function formatOffsetLabel(offsetMinutes: number | null): string {
  if (offsetMinutes == null) return "—";
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMinutes);
  const h = Math.floor(abs / 60);
  const m = abs % 60;
  return `UTC${sign}${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function calendarDayUtc(d: Date): number {
  return Math.floor(d.getTime() / 86400000);
}

function addParsedDate(
  set: Map<number, string>,
  y: number,
  mon: number,
  day: number,
  rawSnippet: string,
): void {
  if (y < 1000 || mon < 1 || mon > 12 || day < 1 || day > 31) return;
  const t = Date.UTC(y, mon - 1, day);
  const chk = new Date(t);
  if (chk.getUTCFullYear() !== y || chk.getUTCMonth() !== mon - 1 || chk.getUTCDate() !== day) return;
  const key = calendarDayUtc(chk);
  if (!set.has(key)) set.set(key, rawSnippet.trim());
}

/**
 * Extract calendar dates from body text (Chinese, English month names, ISO, numeric).
 */
export function extractWrittenDates(text: string): Map<number, string> {
  const byDay = new Map<number, string>();
  if (!text) return byDay;

  for (const m of text.matchAll(/(\d{4})年(\d{1,2})月(\d{1,2})日/g)) {
    addParsedDate(byDay, +m[1], +m[2], +m[3], m[0]);
  }

  for (const m of text.matchAll(
    /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s*(\d{4})\b/gi,
  )) {
    const mon = MONTHS[m[1].toLowerCase()];
    if (mon) addParsedDate(byDay, +m[3], mon, +m[2], m[0]);
  }

  for (const m of text.matchAll(/\b(\d{4})-(\d{2})-(\d{2})\b/g)) {
    addParsedDate(byDay, +m[1], +m[2], +m[3], m[0]);
  }

  for (const m of text.matchAll(/\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b/g)) {
    let a = +m[1];
    let b = +m[2];
    let y = +m[3];
    if (y < 100) y += y >= 70 ? 1900 : 2000;
    if (a > 12) {
      addParsedDate(byDay, y, b, a, m[0]);
    } else if (b > 12) {
      addParsedDate(byDay, y, a, b, m[0]);
    } else {
      addParsedDate(byDay, y, b, a, m[0]);
      if (a !== b) addParsedDate(byDay, y, a, b, m[0]);
    }
  }

  return byDay;
}

function earliestWrittenDay(written: Map<number, string>): { day: number; raw: string } | null {
  if (!written.size) return null;
  let minK = Infinity;
  let raw = "";
  for (const [k, v] of written) {
    if (k < minK) {
      minK = k;
      raw = v;
    }
  }
  return { day: minK, raw };
}

function collectInstitutionHints(text: string): string[] {
  const hints: string[] = [];
  const lower = text.toLowerCase();
  if (INSTITUTIONAL_CUES.test(text)) hints.push("institutional_cues_in_body");
  if (ENGLISH_SPEAKING_COUNTRY_CUES.test(text)) hints.push("english_speaking_region_cues");
  if (CHINA_REGION_CUES.test(text)) hints.push("china_region_cues");
  if (/[\u4e00-\u9fff]/.test(text)) hints.push("contains_cjk_characters");
  return [...new Set(hints)];
}

type RegionBucket = "UK_IE" | "CN_HK_TW" | "EN_WEST" | "JP" | "US_CA_DIVERSE" | "UNKNOWN";

function classifyRegionBucket(text: string): RegionBucket {
  const tl = text.toLowerCase();
  if (CHINA_REGION_CUES.test(text) || /[\u4e00-\u9fff]{3,}/.test(text)) {
    if (/\b(japan|tokyo|osaka|日本|東京)\b/i.test(text)) return "JP";
    return "CN_HK_TW";
  }
  if (/\b(japan|tokyo|osaka|日本)\b/i.test(text)) return "JP";
  if (ENGLISH_SPEAKING_COUNTRY_CUES.test(text)) {
    if (/\.(gov\.uk|ac\.uk)\b/i.test(text) || /\b(england|scotland|wales|northern\s+ireland|britain|uk\b)\b/i.test(tl))
      return "UK_IE";
    if (/\b(united\s+states|u\.s\.a\.?|\busa\b)\b/i.test(tl)) return "US_CA_DIVERSE";
    return "EN_WEST";
  }
  return "UNKNOWN";
}

function offsetAllowedForBucket(bucket: RegionBucket, offsetMinutes: number | null): boolean {
  if (offsetMinutes == null) return true;
  const o = offsetMinutes;
  switch (bucket) {
    case "UK_IE":
      return o >= -60 && o <= 120;
    case "CN_HK_TW":
      return o >= 420 && o <= 540;
    case "JP":
      return o >= 480 && o <= 600;
    case "EN_WEST":
      return o >= -600 && o <= 720;
    case "US_CA_DIVERSE":
      return o >= -600 && o <= 480;
    default:
      return true;
  }
}

function detectPrimaryLanguage(text: string): string {
  const sample = text.slice(0, 12000);
  const cjk = (sample.match(/[\u4e00-\u9fff\u3400-\u4dbf]/g) || []).length;
  const latinLetters = (sample.match(/[a-zA-Z]/g) || []).length;
  const lower = sample.toLowerCase();
  const englishHits = [" the ", " and ", " of ", " to ", " in ", " is ", " for "].filter((w) =>
    lower.includes(w),
  ).length;

  if (cjk > 40 && cjk >= latinLetters * 0.25) return "zh";
  if (latinLetters > 80 && englishHits >= 3) return "en";
  if (latinLetters > 40 && cjk < 15) return "latin_other";
  if (cjk > 15) return "zh_mixed";
  return "undetermined";
}

function normalizeLabeledValue(s: string): string {
  return s.replace(/\s+/g, " ").trim().toLowerCase();
}

export interface UniversalCheckResult {
  flags: ContentFlag[];
  summary: ContentAnalysisSummary;
}

export function runUniversalContentChecks(input: UniversalCheckInput): UniversalCheckResult {
  const text = input.extractedText || "";
  const creator = dnaValue(input.fileDnaRows, "Creator");
  const producer = dnaValue(input.fileDnaRows, "Producer");
  const creationRaw = dnaValue(input.fileDnaRows, "Creation Date");

  const written = extractWrittenDates(text);
  const earliest = earliestWrittenDay(written);
  const pdfParsed = parsePdfMetadataDate(creationRaw);
  const institutionHints = collectInstitutionHints(text);
  const claimsInstitution = INSTITUTIONAL_CUES.test(text) || institutionHints.includes("institutional_cues_in_body");

  const flags: ContentFlag[] = [];

  if (earliest && pdfParsed) {
    const pdfDay = calendarDayUtc(pdfParsed.utc);
    const gap = Math.abs(pdfDay - earliest.day);
    if (gap > 3) {
      const direction =
        pdfDay > earliest.day
          ? `PDF creation is ${gap} calendar days after the earliest written date in the body (${earliest.raw}).`
          : `PDF creation is ${gap} calendar days before the earliest written date in the body (${earliest.raw}).`;
      flags.push({
        type: "date_discrepancy",
        severity: "high",
        label: "Date discrepancy",
        detail: `Document date does not match PDF creation date — ${direction}`,
      });
    }
  }

  if (creator && CONSUMER_CREATOR_RE.test(creator) && claimsInstitution) {
    flags.push({
      type: "consumer_software",
      severity: "high",
      label: "Consumer authoring software",
      detail:
        "Document was created with personal software rather than institutional document systems.",
    });
  }

  const bucket = classifyRegionBucket(text);
  if (bucket !== "UNKNOWN" && pdfParsed?.offsetMinutes != null) {
    if (!offsetAllowedForBucket(bucket, pdfParsed.offsetMinutes)) {
      flags.push({
        type: "timezone_issuer",
        severity: "high",
        label: "Timezone vs issuer",
        detail: "PDF was created in a timezone inconsistent with the claimed issuing institution's location.",
      });
    }
  }

  const placeholderMatches: string[] = [];
  let pm: RegExpExecArray | null;
  const re1 = new RegExp(PLACEHOLDER_AFTER_LABEL_RE.source, PLACEHOLDER_AFTER_LABEL_RE.flags);
  while ((pm = re1.exec(text)) !== null) {
    placeholderMatches.push(pm[0].trim().slice(0, 120));
    if (placeholderMatches.length > 12) break;
  }
  const re2 = new RegExp(BLANK_AFTER_LABEL_RE.source, BLANK_AFTER_LABEL_RE.flags);
  while ((pm = re2.exec(text)) !== null) {
    placeholderMatches.push(pm[0].trim().slice(0, 120));
    if (placeholderMatches.length > 12) break;
  }
  for (const _ of placeholderMatches) {
    flags.push({
      type: "placeholder_field",
      severity: "medium",
      label: "Placeholder or blank field",
      detail: "Unfilled or placeholder field detected — document may be incomplete or templated.",
    });
  }

  const byLabel = new Map<string, Set<string>>();
  let lm: RegExpExecArray | null;
  const labelRe = new RegExp(LABELED_VALUE_RE.source, LABELED_VALUE_RE.flags);
  while ((lm = labelRe.exec(text)) !== null) {
    const key = lm[1].replace(/\s+/g, " ").trim().toLowerCase();
    const val = normalizeLabeledValue(lm[2]);
    if (val.length < 2) continue;
    if (!byLabel.has(key)) byLabel.set(key, new Set());
    byLabel.get(key)!.add(val);
  }
  for (const [label, vals] of byLabel) {
    if (vals.size > 1) {
      flags.push({
        type: "internal_inconsistency",
        severity: "high",
        label: "Internal inconsistency",
        detail: `Internal inconsistency detected — ${label} appears differently across the document.`,
      });
    }
  }

  const lang = detectPrimaryLanguage(text);
  const ukEnglishIssuer =
    /\.(gov\.uk|ac\.uk|nhs\.uk)\b/i.test(text) ||
    /\b(nhs|ukvi|home\s*office|britain|united\s+kingdom)\b/i.test(text);
  const chinaIssuer = CHINA_REGION_CUES.test(text);

  if (ukEnglishIssuer && lang !== "en" && lang !== "undetermined") {
    flags.push({
      type: "language_issuer",
      severity: "high",
      label: "Language vs issuer",
      detail:
        "Document language is inconsistent with the claimed issuing institution's country.",
    });
  }
  if (chinaIssuer && !/[\u4e00-\u9fff]/.test(text) && lang !== "zh" && lang !== "zh_mixed") {
    flags.push({
      type: "language_issuer",
      severity: "high",
      label: "Language vs issuer",
      detail:
        "Document language is inconsistent with the claimed issuing institution's country.",
    });
  }

  const writtenSorted = [...written.entries()].sort((a, b) => a[0] - b[0]).map(([, raw]) => raw);

  const summary: ContentAnalysisSummary = {
    writtenDatesDisplay: writtenSorted,
    primaryLanguage: lang,
    creatorSoftware: creator || "—",
    producerSoftware: producer || "—",
    pdfCreationRaw: creationRaw || "—",
    timezoneLabel: formatOffsetLabel(pdfParsed?.offsetMinutes ?? null),
    institutionHints,
  };

  return { flags, summary };
}
