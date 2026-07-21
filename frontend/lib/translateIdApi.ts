import { getApiBase } from "./analyzeApi";

export interface PlacementItem {
  field_key?: string;
  label?: string;
  translated_value?: string;
  source_bbox?: [number, number, number, number];
  label_bbox?: [number, number, number, number];
  value_bbox?: [number, number, number, number];
  confidence?: number;
  text?: string;
  mode?: string;
  font_size?: number;
}

export interface TranslateIdJsonResponse {
  success: boolean;
  filename: string;
  annotated_image_base64?: string;
  raw_image_base64?: string;
  original_image_base64?: string;
  translated_data: Record<string, any>;
  pdf_base64?: string;
  extracted_regions?: any[];
  bbox_count?: number;
  drawn_count?: number;
  placements?: PlacementItem[];
}

const TRANSLATE_MAX_ATTEMPTS = 3;
const RETRY_DELAY_MS = 15_000;
const FETCH_TIMEOUT_MS = 45_000;

function isNetworkError(error: unknown): boolean {
  if (error instanceof TypeError) return true;
  if (error instanceof Error && /failed to fetch|networkerror|load failed|timeout|aborted/i.test(error.message)) {
    return true;
  }
  return false;
}

/**
 * Uploads an identity document and returns the translated document.
 */
export async function translateForeignIdJson(
  file: File,
  onStatusUpdate?: (status: string) => void
): Promise<TranslateIdJsonResponse> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= TRANSLATE_MAX_ATTEMPTS; attempt++) {
    if (attempt > 1) {
      onStatusUpdate?.("Translating…");
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    }

    const form = new FormData();
    form.append("file", file);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    try {
      onStatusUpdate?.("Translating…");
      const res = await fetch(`${getApiBase()}/api/translate-foreign-id-json`, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        throw new Error("Translation could not be completed.");
      }

      onStatusUpdate?.("Translating…");
      return (await res.json()) as TranslateIdJsonResponse;
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error?.name === "AbortError") {
        lastError = new Error("This file could not be translated. Please try another image or PDF.");
      } else {
        lastError = error;
      }
      if (!isNetworkError(error) || attempt === TRANSLATE_MAX_ATTEMPTS) break;
    }
  }

  if (isNetworkError(lastError)) {
    throw new Error("This file could not be translated. Please try another image or PDF.");
  }
  throw lastError instanceof Error ? lastError : new Error("Translation could not be completed.");
}
