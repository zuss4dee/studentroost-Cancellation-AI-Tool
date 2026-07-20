import { getApiBase } from "./analyzeApi";

export interface TranslateIdJsonResponse {
  success: boolean;
  filename: string;
  translated_data: Record<string, any>;
  pdf_base64: string;
}

const TRANSLATE_MAX_ATTEMPTS = 3;
const RETRY_DELAY_MS = 15_000;

function isNetworkError(error: unknown): boolean {
  if (error instanceof TypeError) return true;
  if (error instanceof Error && /failed to fetch|networkerror|load failed|timeout/i.test(error.message)) {
    return true;
  }
  return false;
}

/**
 * Uploads a foreign ID document and returns both the translated JSON structure
 * and the base64-encoded PDF summary. Automatically retries if server is waking up.
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

    try {
      onStatusUpdate?.("Sending document to Gemini 1.5 Flash for vision translation...");
      const res = await fetch(`${getApiBase()}/api/translate-foreign-id-json`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const errorJson = await res.json().catch(() => ({}));
        const message = errorJson.detail || `Server returned error status ${res.status}`;
        throw new Error(message);
      }

      onStatusUpdate?.("Compiling English PDF report...");
      return (await res.json()) as TranslateIdJsonResponse;
    } catch (error) {
      lastError = error;
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
