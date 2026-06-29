const DEFAULT_API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "https://studentroost-cancellation-ai-tool.onrender.com";

const ANALYZE_MAX_ATTEMPTS = 3;
const RETRY_DELAY_MS = 20_000;

export function getApiBase(): string {
  return DEFAULT_API_BASE.replace(/\/$/, "");
}

/** Best-effort ping so a spun-down Render instance can wake before the user uploads. */
export async function warmAnalysisServer(): Promise<void> {
  const base = getApiBase();
  try {
    const health = await fetch(`${base}/api/health`);
    if (health.ok) return;
  } catch {
    // Fall through to /docs (always available on FastAPI).
  }
  try {
    await fetch(`${base}/docs`);
  } catch {
    // Ignore — warmup is optional.
  }
}

function isNetworkError(error: unknown): boolean {
  if (error instanceof TypeError) return true;
  if (error instanceof Error && /failed to fetch|networkerror|load failed/i.test(error.message)) {
    return true;
  }
  return false;
}

function networkErrorMessage(): string {
  return "Could not reach the analysis server. It may be waking up — wait about a minute and try again.";
}

export async function postAnalyze(
  file: File,
  docTypeKey: string,
  onRetry?: (attempt: number) => void,
): Promise<unknown> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= ANALYZE_MAX_ATTEMPTS; attempt++) {
    if (attempt > 1) {
      onRetry?.(attempt);
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    }

    const form = new FormData();
    form.append("file", file);
    form.append("doc_type_key", docTypeKey);

    try {
      const res = await fetch(`${getApiBase()}/api/analyze`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as { detail?: string }).detail ?? `Request failed: ${res.status}`);
      }
      return res.json();
    } catch (error) {
      lastError = error;
      if (!isNetworkError(error) || attempt === ANALYZE_MAX_ATTEMPTS) break;
    }
  }

  if (isNetworkError(lastError)) {
    throw new Error(networkErrorMessage());
  }
  throw lastError instanceof Error ? lastError : new Error("Upload failed");
}
