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
 * Uploads a foreign ID document to backend, detects foreign text regions & 2D bounding boxes,
 * translates text to English, and returns the annotated image base64 + JSON regions.
 */
export async function generateTranslatedIdOverlay(
  file: File,
  onStatusUpdate?: (status: string) => void
): Promise<OverlayResponse> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= OVERLAY_MAX_ATTEMPTS; attempt++) {
    if (attempt > 1) {
      onStatusUpdate?.(`Analysis server is waking up... Attempt ${attempt} of ${OVERLAY_MAX_ATTEMPTS}`);
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    }

    const form = new FormData();
    form.append("file", file);

    try {
      onStatusUpdate?.("Detecting text regions and 2D bounding boxes with Gemini Vision...");
      const res = await fetch(`${getApiBase()}/api/translated-id-overlay`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const errorJson = await res.json().catch(() => ({}));
        const message = errorJson.detail || `Server returned error status ${res.status}`;
        throw new Error(message);
      }

      onStatusUpdate?.("Rendering English text overlay on ID document...");
      return (await res.json()) as OverlayResponse;
    } catch (error) {
      lastError = error;
      if (!isNetworkError(error) || attempt === OVERLAY_MAX_ATTEMPTS) break;
    }
  }

  if (isNetworkError(lastError)) {
    throw new Error(
      "Could not reach the analysis server. If using Render, it may take ~1 minute to wake up. Please try again."
    );
  }
  throw lastError instanceof Error ? lastError : new Error("ID Overlay processing failed.");
}
