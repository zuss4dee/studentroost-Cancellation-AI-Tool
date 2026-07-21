import { getApiBase } from "./analyzeApi";

export interface ExtractedRegion {
  box_2d: [number, number, number, number];
  original_text: string;
  translated_text: string;
}

export interface OverlayResponse {
  success: boolean;
  filename: string;
  annotated_image_base64: string;
  raw_image_base64: string;
  extracted_regions: ExtractedRegion[];
}

const OVERLAY_MAX_ATTEMPTS = 3;
const RETRY_DELAY_MS = 15_000;

function isNetworkError(error: unknown): boolean {
  if (error instanceof TypeError) return true;
  if (error instanceof Error && /failed to fetch|networkerror|load failed|timeout/i.test(error.message)) {
    return true;
  }
  return false;
}

/**
 * Uploads an identity document to backend and returns the translated document.
 */
export async function generateTranslatedIdOverlay(
  file: File,
  onStatusUpdate?: (status: string) => void
): Promise<OverlayResponse> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= OVERLAY_MAX_ATTEMPTS; attempt++) {
    if (attempt > 1) {
      onStatusUpdate?.("Translating…");
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    }

    const form = new FormData();
    form.append("file", file);

    try {
      onStatusUpdate?.("Translating…");
      const res = await fetch(`${getApiBase()}/api/translated-id-overlay`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        throw new Error("Translation could not be completed.");
      }

      onStatusUpdate?.("Translating…");
      return (await res.json()) as OverlayResponse;
    } catch (error) {
      lastError = error;
      if (!isNetworkError(error) || attempt === OVERLAY_MAX_ATTEMPTS) break;
    }
  }

  if (isNetworkError(lastError)) {
    throw new Error("This file could not be translated. Please try another image or PDF.");
  }
  throw lastError instanceof Error ? lastError : new Error("Translation could not be completed.");
}
