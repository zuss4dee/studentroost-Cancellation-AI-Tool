import { getApiBase } from "./analyzeApi";

export interface TranslateIdJsonResponse {
  success: boolean;
  filename: string;
  translated_data: Record<string, any>;
  pdf_base64: string;
}

/**
 * Uploads a foreign ID document to backend, triggers Gemini 1.5 Flash translation & ReportLab PDF compilation,
 * and automatically triggers browser download of "Translated_ID_Summary.pdf".
 */
export async function translateForeignIdAndDownloadPdf(file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${getApiBase()}/api/translate-foreign-id`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    const message = errorJson.detail || `Server returned error status ${res.status}`;
    throw new Error(message);
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.style.display = "none";
  a.href = url;
  a.download = "Translated_ID_Summary.pdf";
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

/**
 * Uploads a foreign ID document and returns both the translated JSON structure
 * and the base64-encoded PDF summary.
 */
export async function translateForeignIdJson(file: File): Promise<TranslateIdJsonResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${getApiBase()}/api/translate-foreign-id-json`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    const message = errorJson.detail || `Server returned error status ${res.status}`;
    throw new Error(message);
  }

  return (await res.json()) as TranslateIdJsonResponse;
}
