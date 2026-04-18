import "server-only";

import { francAll } from "franc-all";
import { createWorker } from "tesseract.js";
import { fromBuffer } from "pdf2pic";
import { PDFParse } from "pdf-parse";
import { promises as fs } from "fs";
import path from "path";
import os from "os";
import type { PdfExtractionResult, PdfExtractedFields } from "./pdfContentTypes";

export type { PdfExtractionResult, PdfExtractedFields } from "./pdfContentTypes";

/** Multilingual OCR packs requested (English, Chinese variants, Arabic, FR/DE/ES, Hindi, Urdu). */
const OCR_LANGS = "eng+chi_sim+chi_tra+ara+fra+deu+spa+hin+urd";

const MIN_EMBEDDED_TEXT_CHARS = 50;
const MAX_OCR_PAGES = 10;

/** franc ISO 639-3 → readable label (subset + fallback). */
const FRANC_TO_DISPLAY: Record<string, string> = {
  eng: "English",
  cmn: "Chinese (Mandarin)",
  yue: "Chinese (Cantonese)",
  spa: "Spanish",
  fra: "French",
  deu: "German",
  arb: "Arabic",
  hin: "Hindi",
  urd: "Urdu",
  rus: "Russian",
  por: "Portuguese",
  jpn: "Japanese",
  kor: "Korean",
  ita: "Italian",
  nld: "Dutch",
  pol: "Polish",
  tur: "Turkish",
  vie: "Vietnamese",
  tha: "Thai",
  ind: "Indonesian",
  und: "Undetermined",
};

/** franc 639-3 → Google Translate v2 `source` (ISO 639-1 where possible). */
const FRANC_TO_GOOGLE_SOURCE: Record<string, string> = {
  eng: "en",
  cmn: "zh-CN",
  yue: "zh-TW",
  spa: "es",
  fra: "fr",
  deu: "de",
  arb: "ar",
  hin: "hi",
  urd: "ur",
  rus: "ru",
  por: "pt",
  jpn: "ja",
  kor: "ko",
  ita: "it",
  nld: "nl",
  pol: "pl",
  tur: "tr",
  vie: "vi",
  tha: "th",
  ind: "id",
};

function francDistanceToConfidence(d0: number, d1: number | undefined): number {
  if (d1 == null || !Number.isFinite(d1)) return Math.max(0, Math.min(1, 1 - d0 / 4));
  const spread = d1 - d0;
  if (spread <= 0) return 0.35;
  return Math.max(0, Math.min(1, spread / (d1 + 0.25)));
}

function detectLanguage(text: string): { code: string; name: string; confidence: number } {
  const sample = text.slice(0, 12000).trim();
  if (sample.length < 20) {
    return { code: "und", name: FRANC_TO_DISPLAY.und, confidence: 0 };
  }
  const all = francAll(sample);
  if (!all.length) {
    return { code: "und", name: FRANC_TO_DISPLAY.und, confidence: 0 };
  }
  let best = all[0];
  if (best[0] === "und" && all.length > 1) best = all[1];
  const second = all.find((t) => t[0] !== best[0]);
  const conf = francDistanceToConfidence(best[1], second?.[1]);
  const code = best[0];
  const name = FRANC_TO_DISPLAY[code] ?? `Language (${code})`;
  return { code, name, confidence: conf };
}

async function translateToEnglish(text: string, sourceFrancCode: string): Promise<string | null> {
  if (!text.trim()) return "";
  const apiKey = process.env.GOOGLE_TRANSLATE_API_KEY;
  const source = FRANC_TO_GOOGLE_SOURCE[sourceFrancCode] ?? sourceFrancCode.slice(0, 2);

  if (apiKey) {
    try {
      const chunks: string[] = [];
      const max = 4500;
      for (let i = 0; i < text.length; i += max) {
        const q = text.slice(i, i + max);
        const url = `https://translation.googleapis.com/language/translate/v2?key=${encodeURIComponent(apiKey)}`;
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            q,
            target: "en",
            source: sourceFrancCode === "und" ? undefined : source,
            format: "text",
          }),
        });
        if (!res.ok) continue;
        const data = (await res.json()) as {
          data?: { translations?: Array<{ translatedText?: string }> };
        };
        const t = data.data?.translations?.[0]?.translatedText;
        if (t) chunks.push(t);
      }
      if (chunks.length) return chunks.join("\n");
    } catch {
      /* fall through */
    }
    try {
      const { v2 } = await import("@google-cloud/translate");
      const translate = new v2.Translate({ key: apiKey });
      const [translated] = await translate.translate(text, {
        to: "en",
        from: sourceFrancCode === "und" ? undefined : source,
      });
      if (typeof translated === "string") return translated;
      if (Array.isArray(translated)) return (translated as string[]).join("\n");
    } catch {
      /* v2 with API key failed */
    }
  } else {
    try {
      const { v2 } = await import("@google-cloud/translate");
      const translate = new v2.Translate();
      const [translated] = await translate.translate(text, {
        to: "en",
        from: sourceFrancCode === "und" ? undefined : source,
      });
      if (typeof translated === "string") return translated;
      if (Array.isArray(translated)) return (translated as string[]).join("\n");
    } catch {
      /* application default credentials not configured */
    }
  }

  return null;
}

function extractDateStrings(en: string): string[] {
  const found = new Set<string>();
  const add = (s: string) => {
    const t = s.trim();
    if (t.length >= 4) found.add(t);
  };

  for (const m of en.matchAll(/(\d{4})年(\d{1,2})月(\d{1,2})日/g)) add(m[0]);
  for (const m of en.matchAll(
    /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b/gi,
  ))
    add(m[0]);
  for (const m of en.matchAll(/\b\d{4}-\d{2}-\d{2}\b/g)) add(m[0]);
  for (const m of en.matchAll(/\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b/g)) add(m[0]);

  return [...found];
}

const NAME_STOP = new Set(
  "the and for ltd inc plc llc corp company university college hospital department ministry council authority page ref date from subject".split(
    " ",
  ),
);

function extractPersonLikeNames(en: string): string[] {
  const out = new Set<string>();
  const re = /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(en)) !== null) {
    const cand = m[1];
    const parts = cand.toLowerCase().split(/\s+/);
    if (parts.some((p) => NAME_STOP.has(p))) continue;
    if (cand.length < 4) continue;
    out.add(cand);
    if (out.size >= 40) break;
  }
  return [...out];
}

function extractReferenceNumbers(en: string): string[] {
  const out = new Set<string>();
  const re =
    /(?:Ref(?:erence)?|Case|MRN|URN|Tracking|Confirmation|File|Account|ID|No\.?|Number)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_/]{3,})/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(en)) !== null) {
    const v = m[1].trim();
    if (v.length >= 4) out.add(v);
  }
  return [...out];
}

function extractIssuingInstitution(en: string): string | null {
  const labeled =
    /(?:From|Issued\s+by|Issuing\s+(?:authority|body|office)|Prepared\s+by|Sender)\s*[:：]\s*([^\n\r]+)/i.exec(
      en,
    );
  if (labeled?.[1]) return labeled[1].trim().slice(0, 500);

  const lines = en
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  for (let i = 0; i < Math.min(8, lines.length); i++) {
    const line = lines[i];
    if (line.length > 20 && line.length < 200 && /[a-zA-Z]/.test(line)) {
      if (/\b(ltd|limited|university|hospital|ministry|council|authority|department)\b/i.test(line)) {
        return line.slice(0, 500);
      }
    }
  }
  return null;
}

function inferSignatureFilled(en: string): boolean {
  const lines = en.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    if (!/signature/i.test(lines[i])) continue;
    const same = lines[i].replace(/^.*signature\s*[:：]?\s*/i, "").trim();
    if (same.length >= 2 && !/^(n\/a|tbd|none|\.{3,}|—+)$/i.test(same)) return true;
    const next = (lines[i + 1] ?? "").trim();
    if (next.length >= 2 && !/^(n\/a|\.{3,}|—+|_{3,})$/i.test(next)) return true;
    return false;
  }
  return false;
}

function extractFieldsFromEnglish(en: string): PdfExtractedFields {
  return {
    dates: extractDateStrings(en),
    names: extractPersonLikeNames(en),
    referenceNumbers: extractReferenceNumbers(en),
    issuingInstitution: extractIssuingInstitution(en),
    signaturePresent: inferSignatureFilled(en),
  };
}

async function ocrWithPdf2pic(buffer: Buffer, maxPages: number): Promise<string> {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "pdf2pic-"));
  const convert = fromBuffer(buffer, {
    density: 150,
    saveFilename: "page",
    savePath: tmp,
    format: "png",
    width: 2000,
    height: 2000,
  });
  const worker = await createWorker(OCR_LANGS, 1, { logger: () => undefined });
  const parts: string[] = [];
  try {
    for (let p = 1; p <= maxPages; p++) {
      const res = await convert(p, { responseType: "buffer" });
      if (!res.buffer?.length) continue;
      const { data } = await worker.recognize(res.buffer);
      if (data.text?.trim()) parts.push(data.text.trim());
    }
  } finally {
    await worker.terminate();
    await fs.rm(tmp, { recursive: true, force: true }).catch(() => undefined);
  }
  return parts.join("\n\n");
}

async function ocrWithScreenshots(parser: InstanceType<typeof PDFParse>, maxPages: number): Promise<string> {
  const sr = await parser.getScreenshot({
    first: maxPages,
    imageBuffer: true,
    imageDataUrl: false,
    scale: 1.35,
  });
  const worker = await createWorker(OCR_LANGS, 1, { logger: () => undefined });
  const parts: string[] = [];
  try {
    for (const page of sr.pages) {
      if (!page.data?.length) continue;
      const { data } = await worker.recognize(Buffer.from(page.data));
      if (data.text?.trim()) parts.push(data.text.trim());
    }
  } finally {
    await worker.terminate();
  }
  return parts.join("\n\n");
}

/**
 * Layer 1: embedded text via pdf-parse (PDF.js).
 * Layer 2: pdf2pic + tesseract; falls back to pdf-parse screenshots + tesseract if GraphicsMagick/pdf2pic unavailable.
 * Language: franc-all (lingua-style n-gram detector; `@pemistahl/lingua` is not published on npm).
 * Translation: Google Cloud Translate REST (API key) or @google-cloud/translate client when configured.
 */
export async function extractPDFContent(buffer: Buffer): Promise<PdfExtractionResult> {
  const parser = new PDFParse({ data: new Uint8Array(buffer) });
  let rawText = "";
  try {
    const textResult = await parser.getText();
    rawText = (textResult.text ?? "").replace(/\u0000/g, " ").trim();

    if (rawText.length < MIN_EMBEDDED_TEXT_CHARS) {
      const info = await parser.getInfo();
      const pageCount = Math.max(1, info.total || 1);
      const maxPages = Math.min(pageCount, MAX_OCR_PAGES);
      let ocrText = "";
      try {
        ocrText = await ocrWithPdf2pic(buffer, maxPages);
      } catch {
        ocrText = await ocrWithScreenshots(parser, maxPages);
      }
      const combined = ocrText.trim();
      rawText = combined.length >= rawText.length ? combined : rawText;
    }
  } finally {
    await parser.destroy();
  }

  const { code, name, confidence } = detectLanguage(rawText);
  let translatedText = rawText;
  if (code !== "eng" && code !== "und" && rawText.length > 0) {
    const t = await translateToEnglish(rawText, code);
    if (t != null) translatedText = t;
    else translatedText = rawText;
  }

  const extractedFields = extractFieldsFromEnglish(translatedText);

  return {
    rawText,
    translatedText,
    detectedLanguage: name,
    confidence,
    extractedFields,
  };
}
