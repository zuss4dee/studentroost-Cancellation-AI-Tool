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
 * Uploads a foreign ID document and returns the translated overlay image (primary output),
 * original image, extracted English JSON fields, and placement logs.
 * Employs AbortController timeout to prevent infinite loading.
 */
export async function translateForeignIdJson(
  file: File,
  onStatusUpdate?: (status: string) => void
): Promise<TranslateIdJsonResponse> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= TRANSLATE_MAX_ATTEMPTS; attempt++) {
    if (attempt > 1) {
      onStatusUpdate?.(`Analysis server is waking up... Attempt ${attempt} of ${TRANSLATE_MAX_ATTEMPTS}`);
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    }

    const form = new FormData();
    form.append("file", file);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    try {
      onStatusUpdate?.("Detecting foreign text regions & generating English overlay image...");
      const res = await fetch(`${getApiBase()}/api/translate-foreign-id-json`, {
        method: "POST",
        body: form,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        const errorJson = await res.json().catch(() => ({}));
        const message = errorJson.detail || `Server returned error status ${res.status}`;
        throw new Error(message);
      }

      onStatusUpdate?.("Finalizing translated ID document overlay...");
      return (await res.json()) as TranslateIdJsonResponse;
    } catch (error: any) {
      clearTimeout(timeoutId);
      if (error?.name === "AbortError") {
        lastError = new Error("Request timed out. The server took too long to respond. Please try again.");
      } else {
        lastError = error;
      }
      if (!isNetworkError(error) || attempt === TRANSLATE_MAX_ATTEMPTS) break;
    }
  }

  if (isNetworkError(lastError)) {
    throw new Error(
      "Could not reach the analysis server. If using Render, it may take ~1 minute to wake up. Please try again."
    );
  }
  throw lastError instanceof Error ? lastError : new Error("Translation request failed.");
}
