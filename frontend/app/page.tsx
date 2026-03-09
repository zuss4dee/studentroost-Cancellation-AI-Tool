"use client";

import {
  AlertTriangle,
  ChevronRight,
  CloudUpload,
  Dna,
  FileStack,
  FileText,
  FolderOpen,
  Info,
  Play,
  PlusSquare,
  Search,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { supabase } from "../lib/supabaseClient";

const MAX_FILE_SIZE_MB = 200;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const TOAST_AUTO_DISMISS_MS = 6000;

type ToastItem = { id: number; message: string; type: "error" | "success" };

/* Exact colors from design */
const COLORS = {
  pageBg: "#F8F9FA",
  sidebarBg: "#F8F9FA",
  mainBg: "#FFFFFF",
  textPrimary: "#333333",
  textSecondary: "#6C757D",
  purple: "#6B46C1",
  purpleLight: "#E9E3F5",
  border: "#DEE2E6",
  borderDark: "#ADB5BD",
  cardBg: "#FFFFFF",
  cardShadow: "0 1px 3px rgba(0,0,0,0.08)",
  heroGradientFrom: "#5B21B6",
  heroGradientTo: "#312E81",
  greenDot: "#22C55E",
} as const;

const DOC_CLASS_OPTIONS = [
  { value: "visa_refusal", label: "Visa / UKVI Refusal" },
  { value: "medical_letter", label: "Medical Note" },
  { value: "university_letter", label: "University Letter" },
] as const;

type DocTypeKey = (typeof DOC_CLASS_OPTIONS)[number]["value"];

interface PolicyResult {
  verdict: "RED" | "AMBER" | "GREEN";
  reason: string;
  action: string;
  critical_flags_hit?: string[];
}

interface FileDnaRow {
  property: string;
  value: string;
}

interface DetectorSummaryItem {
  flags: string[];
  risk_score?: number;
}

interface AnalyzeResponse {
  filename: string;
  doc_type_key: string;
  forgery_score: number;
  trust_score: number;
  red_flags: string[];
  policy_result: PolicyResult;
  timestamp: string;
  preview_image_base64?: string;
  preview_image_media_type?: string;
  file_dna?: FileDnaRow[];
  ela_image_base64?: string;
  detector_summary?: Record<string, DetectorSummaryItem>;
  ai_confidence?: number;
  ai_indicators?: string[];
  extracted_text?: string;
  noise_findings?: string;
}

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [docTypeKey, setDocTypeKey] = useState<DocTypeKey>("visa_refusal");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [activeView, setActiveView] = useState<"landing" | "result">("landing");
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const toastIdRef = useRef(0);
  const addToast = useCallback((message: string, type: ToastItem["type"] = "error") => {
    const id = ++toastIdRef.current;
    setToasts((prev) => [...prev, { id, message, type }]);
    const t = setTimeout(() => {
      setToasts((prev) => prev.filter((x) => x.id !== id));
    }, TOAST_AUTO_DISMISS_MS);
    return () => clearTimeout(t);
  }, []);
  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((x) => x.id !== id));
  }, []);
  const [recentScans, setRecentScans] = useState<
    { id: string; filename: string; verdict: string | null; created_at: string | null }[]
  >([]);
  const [loadingText, setLoadingText] = useState("Encrypting & Uploading...");

  useEffect(() => {
    let cancelled = false;
    async function loadRecentScans() {
      if (!supabase) return;
      try {
        const { data, error } = await supabase
          .from("scans")
          .select("id, filename, verdict, created_at")
          .order("created_at", { ascending: false })
          .limit(5);
        if (error) {
          // eslint-disable-next-line no-console
          console.error("Failed to fetch recent scans:", error);
          return;
        }
        if (!cancelled && data) {
          setRecentScans(
            data.map((row: any) => ({
              id: String(row.id),
              filename: row.filename ?? "Untitled document",
              verdict: row.verdict ?? null,
              created_at: row.created_at ?? null,
            })),
          );
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error("Unexpected error loading recent scans:", e);
      }
    }
    loadRecentScans();
    return () => {
      cancelled = true;
    };
  }, []);

  const formatTimeAgo = useCallback((iso: string | null | undefined): string => {
    if (!iso) return "Just now";
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return "Just now";
    const diffMs = Date.now() - then;
    const seconds = Math.max(0, Math.floor(diffMs / 1000));
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    if (seconds < 60) return "Just now";
    if (minutes < 60) return `${minutes} min${minutes === 1 ? "" : "s"} ago`;
    if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
    return `${days} day${days === 1 ? "" : "s"} ago`;
  }, []);

  const isLoading = isUploading;

  useEffect(() => {
    if (!isLoading) {
      setLoadingText("Encrypting & Uploading...");
      return;
    }
    setLoadingText("Encrypting & Uploading...");
    const timer = setTimeout(() => {
      setLoadingText("Running Forensic ELA Analysis...");
    }, 2000);
    return () => clearTimeout(timer);
  }, [isLoading]);

  const handleResetScan = useCallback(() => {
    setResult(null);
    setActiveView("landing");
    setIsUploading(false);
    setIsDragging(false);
    setUploadError(null);
    setDocTab("original");
    setDocumentZoom(100);
    setDocumentPan({ x: 0, y: 0 });
    setIsPanning(false);
    setForensicDrawerOpen(false);
    setPolicyDetailsOpen(false);
    setRedFlagsModalOpen(false);
    setExtractedTextModalOpen(false);
  }, []);
  const handleUpload = useCallback(
    async (file: File) => {
      setUploadError(null);
      if (file.size > MAX_FILE_SIZE_BYTES) {
        addToast(`Document too large (Max ${MAX_FILE_SIZE_MB}MB)`, "error");
        return;
      }
      setIsUploading(true);
      setResult(null);
      setActiveView("result");
      const form = new FormData();
      form.append("file", file);
      form.append("doc_type_key", docTypeKey);
      try {
        const res = await fetch("https://studentroost-cancellation-ai-tool.onrender.com/api/analyze", {
          method: "POST",
          body: form,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail ?? `Request failed: ${res.status}`);
        }
        const data: AnalyzeResponse = await res.json();
        setResult(data);
        setActiveView("result");
      } catch (e) {
        const message = e instanceof Error ? e.message : "Upload failed";
        setUploadError(message);
        addToast(message, "error");
      } finally {
        setIsUploading(false);
      }
    },
    [docTypeKey, addToast]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleUpload(file);
    },
    [handleUpload]
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const onFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleUpload(file);
      e.target.value = "";
    },
    [handleUpload]
  );

  const [docTab, setDocTab] = useState<"original" | "ela">("original");
  const [documentZoom, setDocumentZoom] = useState(100);
  const [documentPan, setDocumentPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const [forensicDrawerOpen, setForensicDrawerOpen] = useState(false);
  const [policyDetailsOpen, setPolicyDetailsOpen] = useState(false);
  const [redFlagsModalOpen, setRedFlagsModalOpen] = useState(false);
  const [extractedTextModalOpen, setExtractedTextModalOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [userProfile, setUserProfile] = useState({ displayName: "Sarah Jenkins", role: "Senior Ops Analyst" });
  const [scoreInfoPopover, setScoreInfoPopover] = useState<"forgery" | "trust" | null>(null);
  const policyDetailsAnchorRef = useRef<HTMLButtonElement>(null);

  const profileInitials = userProfile.displayName
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((w) => w[0] ?? "")
    .join("")
    .toUpperCase() || "?";

  const showResults = activeView === "result" && !!result;
  const showSkeleton = isUploading && !result;
  const verdict = result?.policy_result.verdict ?? null;

  const docTypeLabel =
    DOC_CLASS_OPTIONS.find((opt) => opt.value === docTypeKey)?.label ?? "Unknown Type";

  // Simple helper to format the analyzed timestamp similar to the mock
  const analyzedAtLabel =
    result?.timestamp ??
    new Date().toLocaleString(undefined, {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });

  const refId = "REF ID: 892-2021-X";

  // AI confidence: use API value when available, else fallback to forgery_score
  const aiConfidence = result
    ? Math.max(0, Math.min(100, result.ai_confidence ?? result.forgery_score ?? 0))
    : 0;

  // Three verdict stages from scan outcome
  let primaryVerdictTitle = "";
  let secondaryVerdictSubtitle = "";
  if (verdict === "RED") {
    primaryVerdictTitle = "Do not accept";
    secondaryVerdictSubtitle = "Critical policy violation detected";
  } else if (verdict === "AMBER") {
    primaryVerdictTitle = "Further proof required";
    secondaryVerdictSubtitle = "Irregularities require analyst review";
  } else if (verdict === "GREEN") {
    primaryVerdictTitle = "Accept";
    secondaryVerdictSubtitle = "No critical policy violations detected";
  }

  return (
    <div
      className="flex min-h-screen font-sans"
      style={{ backgroundColor: COLORS.pageBg, color: COLORS.textPrimary }}
    >
      {/* Toast notifications - top right */}
      <div
        className="fixed right-4 top-4 z-[100] flex flex-col gap-2"
        role="region"
        aria-label="Notifications"
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            className="flex items-start gap-3 rounded-lg border px-4 py-3 shadow-lg"
            style={{
              backgroundColor: t.type === "error" ? "#FEF2F2" : "#F0FDF4",
              borderColor: t.type === "error" ? "#FECACA" : "#BBF7D0",
              maxWidth: "min(360px, calc(100vw - 2rem))",
            }}
          >
            <span
              className="text-[13px] font-medium"
              style={{ color: t.type === "error" ? "#B91C1C" : "#166534", flex: 1 }}
            >
              {t.type === "error" ? "Error: " : ""}
              {t.message}
            </span>
            <button
              type="button"
              onClick={() => dismissToast(t.id)}
              className="shrink-0 rounded p-1 hover:opacity-80"
              style={{ color: t.type === "error" ? "#B91C1C" : "#166534" }}
              aria-label="Dismiss"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        ))}
      </div>

      {/* Left sidebar - fixed width ~25% */}
      <aside
        className="flex w-[280px] shrink-0 flex-col border-r p-5"
        style={{ backgroundColor: COLORS.sidebarBg, borderColor: COLORS.border }}
      >
        {/* Logo: purple P / folded paper + Student Roost */}
        <div className="mb-8">
          <div className="flex items-center gap-2">
            <div
              className="flex h-9 w-9 items-center justify-center rounded-lg text-white"
              style={{ backgroundColor: COLORS.purple }}
            >
              <FileText className="h-5 w-5" strokeWidth={2} />
            </div>
            <div>
              <div className="text-[15px] font-bold" style={{ color: COLORS.textPrimary }}>
                Student Roost
              </div>
              <div className="text-[11px] font-medium uppercase tracking-wide" style={{ color: COLORS.textSecondary }}>
                FRAUD OPS
              </div>
            </div>
          </div>
        </div>

        {/* DOCUMENT CLASS */}
        <div className="mb-5">
          <div
            className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider"
            style={{ color: COLORS.textSecondary }}
          >
            Document Class
          </div>
          <select
            value={docTypeKey}
            onChange={(e) => setDocTypeKey(e.target.value as DocTypeKey)}
            className="w-full cursor-pointer appearance-none rounded-md border bg-[#F1F3F5] py-2.5 pl-3 pr-8 text-[13px] outline-none focus:ring-1"
            style={{
              borderColor: COLORS.borderDark,
              color: COLORS.textPrimary,
            }}
          >
            {DOC_CLASS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* NEW ANALYSIS - dashed purple border, purple "or click to browse" */}
        <div className="mb-5">
          <div
            className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider"
            style={{ color: COLORS.textSecondary }}
          >
            New Analysis
          </div>
          <label
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed py-7 transition-colors ${
              isUploading ? "pointer-events-none opacity-60" : ""
            }`}
            style={{
              backgroundColor: "#F1F3F5",
              borderColor: isDragging ? COLORS.purple : COLORS.purple,
              opacity: isDragging ? 0.9 : 1,
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.jpg,.jpeg,.png,.tiff,.tif"
              className="hidden"
              onChange={onFileInputChange}
              disabled={isUploading}
            />
            <CloudUpload className="mb-2 h-9 w-9" style={{ color: COLORS.purple }} />
            <span className="text-[13px] font-semibold" style={{ color: COLORS.textPrimary }}>
              Drop PDF to Scan
            </span>
            <span className="mt-0.5 text-[11px]" style={{ color: COLORS.purple }}>
              or click to browse
            </span>
          </label>
          {uploadError && (
            <p className="mt-1.5 text-[11px]" style={{ color: "#DC2626" }}>{uploadError}</p>
          )}
        </div>

        {/* MENU - Current Analysis with light purple bg + purple text */}
        <div className="mb-5">
          <div
            className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider"
            style={{ color: COLORS.textSecondary }}
          >
            Menu
          </div>
          <nav className="flex flex-col gap-0.5">
            <a
              href="#"
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-[13px] font-medium"
              style={{ backgroundColor: COLORS.purpleLight, color: COLORS.purple }}
            >
              <Search className="h-4 w-4" style={{ color: COLORS.purple }} />
              Current Analysis
            </a>
            <a
              href="#"
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-[13px] font-medium hover:bg-white/80"
              style={{ color: COLORS.textPrimary }}
            >
              <FolderOpen className="h-4 w-4" style={{ color: COLORS.textSecondary }} />
              Case History
            </a>
          </nav>
        </div>

        {/* RECENT SCANS */}
        <div className="mt-auto">
          <div
            className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider"
            style={{ color: COLORS.textSecondary }}
          >
            Recent Scans
          </div>
          {recentScans.length === 0 ? (
            <p className="text-[11px]" style={{ color: COLORS.textSecondary }}>
              No scans yet. Your last 5 scans will appear here.
            </p>
          ) : (
            <ul className="space-y-0.5">
              {recentScans.map((scan) => {
                const verdict = (scan.verdict || "").toUpperCase();
                const color =
                  verdict === "RED"
                    ? "#EF4444"
                    : verdict === "GREEN"
                      ? "#22C55E"
                      : verdict === "AMBER"
                        ? "#F59E0B"
                        : "#9CA3AF"; // default gray
                return (
                  <li
                    key={scan.id}
                    className="flex items-center gap-2 rounded-lg px-2 py-1.5 transition-colors transition-shadow hover:bg-slate-50 hover:shadow-sm"
                  >
                    <FileText className="h-4 w-4 shrink-0" style={{ color: COLORS.textSecondary }} />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px]" style={{ color: COLORS.textPrimary }}>
                        {scan.filename}
                      </div>
                      <div className="text-[11px]" style={{ color: COLORS.textSecondary }}>
                        {formatTimeAgo(scan.created_at)}
                      </div>
                    </div>
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: color }}
                      title={verdict || "Unknown"}
                    />
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* User profile - clickable to open profile drawer */}
        <div
          className="mt-5 flex items-center gap-3 border-t pt-4"
          style={{ borderColor: COLORS.border }}
        >
          <button
            type="button"
            onClick={() => setProfileOpen(true)}
            className="flex min-w-0 flex-1 items-center gap-3 rounded-lg text-left transition-colors hover:bg-white/60"
          >
            <div
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-[12px] font-semibold"
              style={{ backgroundColor: COLORS.purpleLight, color: COLORS.purple }}
            >
              {profileInitials}
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13px] font-medium" style={{ color: COLORS.textPrimary }}>
                {userProfile.displayName}
              </div>
              <div className="truncate text-[11px]" style={{ color: COLORS.textSecondary }}>
                {userProfile.role}
              </div>
            </div>
          </button>
          <button
            type="button"
            onClick={() => setProfileOpen(true)}
            className="rounded border p-1 hover:opacity-80"
            style={{ borderColor: COLORS.border, color: COLORS.textSecondary }}
            title="Profile"
          >
            <PlusSquare className="h-4 w-4" />
          </button>
        </div>
      </aside>

      {/* Main content - white background */}
      <main className="flex flex-1 flex-col" style={{ backgroundColor: COLORS.mainBg }}>
        {/* Header bar - Landing / Result on far right */}
        <div
          className="flex items-center justify-end gap-1 border-b px-6 py-2.5"
          style={{ borderColor: COLORS.border }}
        >
          <button
            type="button"
            className="rounded-full px-4 py-1.5 text-[13px] font-medium transition"
            style={
              activeView === "landing"
                ? { backgroundColor: COLORS.purple, color: "#FFFFFF" }
                : { color: COLORS.textSecondary, backgroundColor: "transparent" }
            }
          >
            Landing
          </button>
          <button
            type="button"
            className="rounded-full px-4 py-1.5 text-[13px] font-medium transition disabled:opacity-50"
            style={
              activeView === "result"
                ? { backgroundColor: COLORS.purple, color: "#FFFFFF" }
                : { color: COLORS.textSecondary, backgroundColor: "transparent" }
            }
          >
            Result
          </button>
        </div>

        <div className="flex-1 overflow-auto p-6">
          {!showResults ? (
            /* Landing: hero + feature cards */
            <>
              {/* Hero - dark purple gradient, white text, white buttons with purple text */}
              <div
                className="mx-auto max-w-4xl rounded-2xl p-8 text-white"
                style={{
                  background: `linear-gradient(to bottom, ${COLORS.heroGradientFrom}, ${COLORS.heroGradientTo})`,
                  boxShadow: "0 4px 14px rgba(0,0,0,0.15)",
                }}
              >
                <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-white/15 px-3 py-1.5">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: COLORS.greenDot, boxShadow: `0 0 0 3px ${COLORS.greenDot}40` }}
                  />
                  <span className="text-[12px] font-medium opacity-95">
                    System Online v4.2.0 Active
                  </span>
                </div>
                <h1 className="text-[28px] font-bold leading-tight tracking-tight">
                  Automated Document Forensics Engine
                </h1>
                <p className="mt-2 max-w-xl text-[14px] leading-relaxed opacity-95">
                  Instantly detect metadata anomalies, pixel manipulation, and AI-generated text in
                  uploaded compliance documents.
                </p>
                <div className="mt-6 flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-[13px] font-semibold shadow-md transition hover:opacity-95"
                    style={{ color: COLORS.purple }}
                  >
                    <CloudUpload className="h-4 w-4" />
                    Start New Scan
                  </button>
                  <button
                    type="button"
                    className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-[13px] font-medium shadow-md transition hover:opacity-95"
                    style={{ color: COLORS.purple }}
                  >
                    <Play className="h-4 w-4" style={{ color: COLORS.purple }} />
                    View Demo
                  </button>
                </div>
              </div>

              {/* Feature cards - white, rounded, shadow, purple icons */}
              <div className="mx-auto mt-6 grid max-w-4xl grid-cols-1 gap-5 sm:grid-cols-3">
                <div
                  className="rounded-xl p-5"
                  style={{ backgroundColor: COLORS.cardBg, boxShadow: COLORS.cardShadow }}
                >
                  <div
                    className="mb-3 flex h-10 w-10 items-center justify-center rounded-full"
                    style={{ backgroundColor: COLORS.purpleLight }}
                  >
                    <FileText className="h-5 w-5" style={{ color: COLORS.purple }} />
                  </div>
                  <h3 className="text-[14px] font-bold" style={{ color: COLORS.textPrimary }}>
                    File DNA & Metadata
                  </h3>
                  <p className="mt-1.5 text-[13px] leading-relaxed" style={{ color: COLORS.textSecondary }}>
                    Deep analysis of EXIF data, creation dates, and modification history to identify
                    inconsistencies in file origin.
                  </p>
                </div>
                <div
                  className="rounded-xl p-5"
                  style={{ backgroundColor: COLORS.cardBg, boxShadow: COLORS.cardShadow }}
                >
                  <div
                    className="mb-3 flex h-10 w-10 items-center justify-center rounded-full"
                    style={{ backgroundColor: COLORS.purpleLight }}
                  >
                    <FileStack className="h-5 w-5" style={{ color: COLORS.purple }} />
                  </div>
                  <h3 className="text-[14px] font-bold" style={{ color: COLORS.textPrimary }}>
                    Pixel & Layout Analysis
                  </h3>
                  <p className="mt-1.5 text-[13px] leading-relaxed" style={{ color: COLORS.textSecondary }}>
                    Error Level Analysis (ELA) to spot digital tampering, copy-paste artifacts, and
                    font inconsistencies invisible to the naked eye.
                  </p>
                </div>
                <div
                  className="rounded-xl p-5"
                  style={{ backgroundColor: COLORS.cardBg, boxShadow: COLORS.cardShadow }}
                >
                  <div
                    className="mb-3 flex h-10 w-10 items-center justify-center rounded-full"
                    style={{ backgroundColor: COLORS.purpleLight }}
                  >
                    <span className="text-[18px] font-bold" style={{ color: COLORS.purple }}>%</span>
                  </div>
                  <h3 className="text-[14px] font-bold" style={{ color: COLORS.textPrimary }}>
                    AI Content Detection
                  </h3>
                  <p className="mt-1.5 text-[13px] leading-relaxed" style={{ color: COLORS.textSecondary }}>
                    Advanced NLP models trained to distinguish between human-written narratives and
                    LLM-generated fabrication in personal statements.
                  </p>
                </div>
              </div>

              <p
                className="mt-8 text-center text-[11px]"
                style={{ color: COLORS.textSecondary }}
              >
                © 2024 Student Roost Operations. Internal Use Only.
              </p>
            </>
          ) : showSkeleton ? (
            /* Skeleton: gray pulsing replica of result layout */
            <div className="mx-auto flex max-w-6xl items-start gap-6 animate-pulse">
              <section className="flex-1 space-y-4">
                <header className="flex items-start justify-between gap-4">
                  <div className="space-y-2">
                    <div className="h-3 w-24 rounded bg-slate-200" />
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="h-5 w-64 rounded bg-slate-200" />
                      <div className="h-6 w-20 rounded-full bg-slate-200" />
                    </div>
                  </div>
                  <div className="h-4 w-28 rounded bg-slate-200" />
                </header>
                <div
                  className="rounded-2xl border bg-white"
                  style={{ borderColor: COLORS.border, boxShadow: "0 8px 24px rgba(15,23,42,0.06)" }}
                >
                  <div className="flex items-center justify-between border-b px-6 py-3" style={{ borderColor: COLORS.border }}>
                    <div className="h-4 w-36 rounded bg-slate-200" />
                    <div className="flex gap-2">
                      <div className="h-7 w-20 rounded-full bg-slate-200" />
                      <div className="h-7 w-24 rounded-full bg-slate-200" />
                    </div>
                  </div>
                  <div className="flex flex-col items-center bg-[#F5F5F7] px-10 py-8">
                    <div
                      className="flex aspect-[3/4] w-full max-w-xl items-center justify-center overflow-hidden rounded-xl border bg-slate-200"
                      style={{ borderColor: COLORS.border }}
                    />
                    <div className="mt-4 flex gap-2">
                      <div className="h-8 w-24 rounded-full bg-slate-200" />
                      <div className="h-8 w-12 rounded-full bg-slate-200" />
                    </div>
                  </div>
                </div>
              </section>
              <aside className="w-[400px] space-y-4">
                <div
                  className="rounded-2xl border bg-white p-6"
                  style={{ borderColor: COLORS.border, boxShadow: COLORS.cardShadow }}
                >
                  <div className="mb-5 h-3 w-28 rounded bg-slate-200" />
                  <div className="rounded-xl bg-slate-200 px-5 py-4">
                    <div className="h-6 w-40 rounded bg-slate-300" />
                    <div className="mt-2 h-4 w-full rounded bg-slate-300" />
                    <div className="mt-3 h-6 w-20 rounded-full bg-slate-300" />
                  </div>
                  <div className="mt-5 grid grid-cols-2 gap-3">
                    <div className="rounded-lg border bg-slate-100 px-3 py-2.5" style={{ borderColor: COLORS.border }}>
                      <div className="h-3 w-20 rounded bg-slate-200" />
                      <div className="mt-2 h-6 w-12 rounded bg-slate-200" />
                      <div className="mt-2 h-1.5 w-full rounded-full bg-slate-200" />
                    </div>
                    <div className="rounded-lg border bg-slate-100 px-3 py-2.5" style={{ borderColor: COLORS.border }}>
                      <div className="h-3 w-16 rounded bg-slate-200" />
                      <div className="mt-2 h-6 w-12 rounded bg-slate-200" />
                      <div className="mt-2 h-1.5 w-full rounded-full bg-slate-200" />
                    </div>
                  </div>
                </div>
                <div
                  className="rounded-2xl border bg-white p-5"
                  style={{ borderColor: COLORS.border, boxShadow: COLORS.cardShadow }}
                >
                  <div className="mb-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="h-6 w-6 rounded-full bg-slate-200" />
                      <div className="h-4 w-32 rounded bg-slate-200" />
                    </div>
                    <div className="h-5 w-16 rounded-full bg-slate-200" />
                  </div>
                  <div className="space-y-2">
                    <div className="h-4 w-full rounded bg-slate-200" />
                    <div className="h-4 w-11/12 rounded bg-slate-200" />
                    <div className="h-4 w-4/5 rounded bg-slate-200" />
                  </div>
                </div>
                <div className="flex justify-end">
                  <div className="h-10 w-36 rounded-lg bg-slate-200" />
                </div>
              </aside>
            </div>
          ) : (
            /* Results view – two-column layout matching case analysis UI */
            <div className="mx-auto flex max-w-6xl items-start gap-6">
              {/* Left column: case header + document evidence card */}
              <section className="flex-1 space-y-4">
                {/* Case header */}
                <header className="flex items-start justify-between gap-4">
                  <div>
                    <div
                      className="text-[11px] font-semibold uppercase tracking-wider"
                      style={{ color: COLORS.textSecondary }}
                    >
                      {refId}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-2">
                      <h1 className="text-[18px] font-semibold" style={{ color: COLORS.textPrimary }}>
                        Case File Analysis:{" "}
                        <span className="font-normal" style={{ color: COLORS.textSecondary }}>
                          {result!.filename}
                        </span>
                      </h1>
                      <span
                        className="rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-wide"
                        style={{ backgroundColor: COLORS.purpleLight, color: COLORS.purple }}
                      >
                        {docTypeLabel}
                      </span>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2 text-right">
                    <div className="text-[11px]" style={{ color: COLORS.textSecondary }}>
                      Analyzed:{" "}
                      <span className="font-medium" style={{ color: COLORS.textPrimary }}>
                        {analyzedAtLabel}
                      </span>
                    </div>
                  </div>
                </header>

                {/* Document evidence card (static preview placeholder) */}
                <div
                  className="rounded-2xl border bg-white"
                  style={{ borderColor: COLORS.border, boxShadow: "0 8px 24px rgba(15,23,42,0.06)" }}
                >
                  <div
                    className="flex items-center justify-between border-b px-6 py-3"
                    style={{ borderColor: COLORS.border }}
                  >
                    <div className="text-[13px] font-semibold" style={{ color: COLORS.textPrimary }}>
                      Document Evidence
                    </div>
                    <div className="flex items-center gap-1 text-[12px] font-medium">
                      {[
                        { id: "original" as const, label: "Original" },
                        { id: "ela" as const, label: "ELA Heatmap" },
                      ].map((tab) => {
                        const isActive = docTab === tab.id;
                        return (
                          <button
                            key={tab.id}
                            type="button"
                            onClick={() => setDocTab(tab.id)}
                            className={`rounded-full px-3 py-1 transition ${
                              isActive ? "bg-slate-100 text-slate-900" : "text-slate-500 hover:bg-slate-100"
                            }`}
                          >
                            {tab.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  <div className="flex flex-col items-center bg-[#F5F5F7] px-10 py-8">
                    {docTab === "original" && (
                      <>
                        <div
                          className="flex aspect-[3/4] w-full max-w-md items-center justify-center overflow-hidden rounded-xl border bg-white"
                          style={{
                            borderColor: COLORS.border,
                            boxShadow: "0 12px 30px rgba(0,0,0,0.12)",
                            cursor: isPanning ? "grabbing" : "grab",
                          }}
                          onMouseDown={(e) => {
                            if (e.button !== 0) return;
                            setIsPanning(true);
                            panStartRef.current = {
                              x: e.clientX,
                              y: e.clientY,
                              panX: documentPan.x,
                              panY: documentPan.y,
                            };
                          }}
                          onMouseMove={(e) => {
                            if (!panStartRef.current) return;
                            setDocumentPan({
                              x: panStartRef.current.panX + (e.clientX - panStartRef.current.x),
                              y: panStartRef.current.panY + (e.clientY - panStartRef.current.y),
                            });
                          }}
                          onMouseUp={() => {
                            setIsPanning(false);
                            panStartRef.current = null;
                          }}
                          onMouseLeave={() => {
                            if (panStartRef.current) {
                              setIsPanning(false);
                              panStartRef.current = null;
                            }
                          }}
                        >
                          <div
                            className="flex h-full w-full items-center justify-center select-none"
                            style={{
                              transform: `translate(${documentPan.x}px, ${documentPan.y}px) scale(${documentZoom / 100})`,
                              transformOrigin: "center center",
                            }}
                            onMouseDown={(e) => e.preventDefault()}
                          >
                            {result?.preview_image_base64 ? (
                              <img
                                src={`data:${result.preview_image_media_type ?? "image/png"};base64,${result.preview_image_base64}`}
                                alt="Document preview"
                                className="h-full w-full object-contain pointer-events-none"
                                draggable={false}
                              />
                            ) : (
                              <div className="h-full w-full bg-[url('/sample-case-document.png')] bg-cover bg-center pointer-events-none" />
                            )}
                          </div>
                        </div>
                        {/* Zoom & pan control */}
                        <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
                          <div className="inline-flex items-center gap-2 rounded-full bg-[#111827]/90 px-3 py-1 text-[11px] text-white shadow-md">
                            <button
                              type="button"
                              onClick={() => setDocumentZoom((z) => Math.max(50, z - 10))}
                              className="flex h-6 w-6 items-center justify-center rounded-full bg-black/40 text-[14px] hover:bg-black/60"
                            >
                              -
                            </button>
                            <span className="min-w-[2.5rem] px-1 text-center">{documentZoom}%</span>
                            <button
                              type="button"
                              onClick={() => setDocumentZoom((z) => Math.min(200, z + 10))}
                              className="flex h-6 w-6 items-center justify-center rounded-full bg-black/40 text-[14px] hover:bg-black/60"
                            >
                              +
                            </button>
                          </div>
                          <button
                            type="button"
                            onClick={() => setDocumentPan({ x: 0, y: 0 })}
                            className="rounded-full bg-[#111827]/90 px-3 py-1 text-[11px] text-white shadow-md hover:bg-[#111827]"
                          >
                            Reset pan
                          </button>
                        </div>
                      </>
                    )}

                    {docTab === "ela" && (
                      <div className="flex w-full max-w-md flex-col items-center">
                        <div
                          className="flex aspect-[3/4] w-full max-w-md items-center justify-center overflow-hidden rounded-xl border bg-white"
                          style={{ borderColor: COLORS.border, boxShadow: "0 12px 30px rgba(0,0,0,0.12)" }}
                        >
                          {result?.ela_image_base64 ? (
                            <img
                              src={`data:image/png;base64,${result.ela_image_base64}`}
                              alt="ELA heatmap"
                              className="h-full w-full object-contain"
                            />
                          ) : (
                            <div
                              className="flex h-64 w-full items-center justify-center rounded-xl border bg-gradient-to-br from-slate-900 via-rose-500 to-amber-400 text-[12px] font-medium text-white"
                              style={{ borderColor: COLORS.border }}
                            >
                              ELA heatmap – bright regions may indicate digital manipulation.
                            </div>
                          )}
                        </div>
                        <p className="mt-3 text-center text-[11px] text-slate-600">
                          Error Level Analysis: dark areas are consistent, bright areas highlight potential edits.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </section>

              {/* Right column: verdict and red flags */} 
              <aside className="w-[400px] space-y-4">
                {/* Forensic verdict card – matches FORENSIC VERDICT design */}
                <div
                  className="rounded-2xl border bg-white p-6"
                  style={{ borderColor: COLORS.border, boxShadow: COLORS.cardShadow }}
                >
                  <div
                    className="mb-5 text-[11px] font-semibold uppercase tracking-wider"
                    style={{ color: COLORS.textSecondary }}
                  >
                    FORENSIC VERDICT
                  </div>
                  <div
                    className="relative w-full rounded-xl px-5 py-4 flex items-start gap-3"
                    style={{
                      backgroundColor:
                        verdict === "RED" ? "#FEE2E2" : verdict === "AMBER" ? "#FEF3C7" : "#DCFCE7",
                      color: verdict === "RED" ? "#B91C1C" : verdict === "AMBER" ? "#92400E" : "#166534",
                    }}
                  >
                    <AlertTriangle className="h-8 w-8 shrink-0 mt-0.5" strokeWidth={2.25} aria-hidden />
                    <div className="flex-1 min-w-0">
                      <div className="text-[17px] font-bold uppercase tracking-wide leading-tight">
                        {primaryVerdictTitle}
                      </div>
                      <p className="mt-1.5 text-[12px] font-normal leading-snug">
                        {result?.policy_result?.reason ?? secondaryVerdictSubtitle}
                      </p>
                      <div className="mt-3">
                        <button
                          ref={policyDetailsAnchorRef}
                          type="button"
                          onClick={() => setPolicyDetailsOpen((o) => !o)}
                          className="inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[10px] font-medium opacity-90 hover:opacity-100"
                          style={{
                            borderColor: "currentColor",
                            color: "inherit",
                          }}
                        >
                          <Info className="h-3 w-3" />
                          See Details
                        </button>
                      </div>
                    </div>
                    {policyDetailsOpen && (
                      <>
                        <div
                          className="fixed inset-0 z-10 bg-black/20"
                          aria-hidden
                          onClick={() => setPolicyDetailsOpen(false)}
                        />
                        <div
                          className="fixed left-1/2 top-1/2 z-20 w-[90vw] max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-xl border bg-white shadow-xl"
                          style={{ borderColor: COLORS.border }}
                        >
                          <div className="flex items-center justify-between border-b px-4 py-3" style={{ borderColor: COLORS.border }}>
                            <span className="text-[13px] font-semibold" style={{ color: COLORS.textPrimary }}>
                              Policy details
                            </span>
                            <button
                              type="button"
                              onClick={() => setPolicyDetailsOpen(false)}
                              className="rounded p-1 hover:bg-slate-100"
                              aria-label="Close"
                            >
                              <X className="h-4 w-4" style={{ color: COLORS.textSecondary }} />
                            </button>
                          </div>
                          <div className="max-h-[60vh] overflow-y-auto p-4 space-y-4">
                            <section>
                              <h4 className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: COLORS.textSecondary }}>
                                Recommended action
                              </h4>
                              <p className="mt-1.5 text-[13px] leading-relaxed" style={{ color: COLORS.textPrimary }}>
                                {result?.policy_result?.action ?? "—"}
                              </p>
                            </section>
                            {(result?.policy_result?.critical_flags_hit?.length ?? 0) > 0 && (
                              <section className="pt-3 border-t" style={{ borderColor: COLORS.border }}>
                                <h4 className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: COLORS.textSecondary }}>
                                  Critical flags hit
                                </h4>
                                <ul className="mt-1.5 space-y-1 list-none pl-0 text-[12px] leading-relaxed" style={{ color: COLORS.textPrimary }}>
                                  {result!.policy_result!.critical_flags_hit!.map((f, i) => (
                                    <li key={i} className="flex items-start gap-2">
                                      <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-400" />
                                      <span>{f}</span>
                                    </li>
                                  ))}
                                </ul>
                              </section>
                            )}
                          </div>
                        </div>
                      </>
                    )}
                  </div>

                  <div className="mt-5 grid grid-cols-2 gap-3 min-w-0">
                    <div className="min-w-0 flex flex-col rounded-lg border bg-white px-3 py-2.5 text-left" style={{ borderColor: COLORS.border }}>
                      <div
                        className="flex items-center gap-1 text-[9px] font-semibold uppercase tracking-wider leading-tight"
                        style={{ color: COLORS.textSecondary }}
                      >
                        <span>Forgery Probability</span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.preventDefault();
                            setScoreInfoPopover((v) => (v === "forgery" ? null : "forgery"));
                          }}
                          className="shrink-0 cursor-help rounded p-0.5 opacity-70 hover:opacity-100 focus:outline-none focus:ring-1 focus:ring-offset-0"
                          style={{ color: COLORS.textSecondary }}
                          title="What does Forgery Probability mean?"
                          aria-label="Explain forgery probability"
                        >
                          <Info className="h-3 w-3" />
                        </button>
                      </div>
                      <div className="mt-2 flex w-full items-baseline gap-0.5">
                        <span className="shrink-0 text-[18px] font-bold tabular-nums" style={{ color: COLORS.textPrimary }}>
                          {result!.forgery_score}
                        </span>
                        <span className="shrink-0 text-[10px] font-normal" style={{ color: COLORS.textSecondary }}>
                          /100
                        </span>
                      </div>
                      <div className="mt-2 h-1.5 w-full rounded-full bg-slate-200 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-red-500 transition-[width]"
                          style={{ width: `${Math.min(100, Math.max(0, result!.forgery_score))}%` }}
                        />
                      </div>
                    </div>
                    <div className="min-w-0 flex flex-col rounded-lg border bg-white px-3 py-2.5 text-left" style={{ borderColor: COLORS.border }}>
                      <div
                        className="flex items-center gap-1 text-[9px] font-semibold uppercase tracking-wider leading-tight"
                        style={{ color: COLORS.textSecondary }}
                      >
                        <span>Trust Score</span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.preventDefault();
                            setScoreInfoPopover((v) => (v === "trust" ? null : "trust"));
                          }}
                          className="shrink-0 cursor-help rounded p-0.5 opacity-70 hover:opacity-100 focus:outline-none focus:ring-1 focus:ring-offset-0"
                          style={{ color: COLORS.textSecondary }}
                          title="What does Trust Score mean?"
                          aria-label="Explain trust score"
                        >
                          <Info className="h-3 w-3" />
                        </button>
                      </div>
                      <div className="mt-2 flex w-full items-baseline gap-0.5">
                        <span className="shrink-0 text-[18px] font-bold tabular-nums" style={{ color: COLORS.textPrimary }}>
                          {result!.trust_score}
                        </span>
                        <span className="shrink-0 text-[10px] font-normal" style={{ color: COLORS.textSecondary }}>
                          /100
                        </span>
                      </div>
                      <div className="mt-2 h-1.5 w-full rounded-full bg-slate-200 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-green-500 transition-[width]"
                          style={{ width: `${Math.min(100, Math.max(0, result!.trust_score))}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Score info popover (click-to-show) */}
                  {scoreInfoPopover && (
                    <div
                      className="mt-3 rounded-lg border px-3 py-2.5 text-left"
                      style={{ borderColor: COLORS.border, backgroundColor: COLORS.purpleLight }}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-[12px] leading-snug" style={{ color: COLORS.textPrimary }}>
                          {scoreInfoPopover === "forgery" ? (
                            <>Forgery probability means the likelihood that the document has been digitally altered or fabricated—through metadata changes, pixel manipulation, or content edits. Higher score = more signs of forgery.</>
                          ) : (
                            <>Trust score means how reliable the document appears based on consistency of metadata, file structure, and forensic analysis. Higher score = more trustworthy.</>
                          )}
                        </p>
                        <button
                          type="button"
                          onClick={() => setScoreInfoPopover(null)}
                          className="shrink-0 rounded p-0.5 hover:bg-white/50"
                          aria-label="Close"
                        >
                          <X className="h-4 w-4" style={{ color: COLORS.textSecondary }} />
                        </button>
                      </div>
                    </div>
                  )}

                </div>

                {/* Critical red flags & AI confidence */} 
                <div
                  className="rounded-2xl border bg-white p-5"
                  style={{ borderColor: COLORS.border, boxShadow: COLORS.cardShadow }}
                >
                  <div className="mb-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-red-100 text-red-600 text-[13px]">
                        !
                      </span>
                      <div className="text-[12px] font-semibold" style={{ color: COLORS.textPrimary }}>
                        Critical Red Flags
                      </div>
                    </div>
                    <span className="rounded-full bg-red-100 px-2 py-0.5 text-[11px] font-medium text-red-700">
                      {result!.red_flags.length} Issues
                    </span>
                  </div>

                  {result!.red_flags.length === 0 ? (
                    <p className="text-[13px]" style={{ color: COLORS.textSecondary }}>
                      No red flags detected.
                    </p>
                  ) : (
                    <>
                      <ul className="space-y-2">
                        {result!.red_flags.slice(0, 3).map((flag, index) => (
                          <li
                            key={index}
                            className="flex items-start gap-2 text-[13px]"
                            style={{ color: COLORS.textPrimary }}
                          >
                            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-400" />
                            <span>{flag}</span>
                          </li>
                        ))}
                      </ul>
                      {result!.red_flags.length > 3 && (
                        <button
                          type="button"
                          onClick={() => setRedFlagsModalOpen(true)}
                          className="mt-2 text-[12px] font-medium text-red-600 hover:underline"
                        >
                          Show All ({result!.red_flags.length})
                        </button>
                      )}
                    </>
                  )}
                </div>

                {/* Toggle for forensic report drawer + reset button */}
                <div className="mt-1 flex justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => setForensicDrawerOpen(true)}
                    className="inline-flex items-center gap-1 rounded-lg border px-3 py-2 text-[12px] font-medium transition hover:bg-slate-50"
                    style={{ borderColor: COLORS.border, color: COLORS.textPrimary }}
                  >
                    <FileStack className="h-4 w-4" />
                    Forensic Report
                    <ChevronRight className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={handleResetScan}
                    className="inline-flex items-center gap-1 rounded-lg border px-3 py-2 text-[12px] font-medium transition hover:bg-slate-50"
                    style={{ borderColor: COLORS.border, color: COLORS.purple }}
                  >
                    <PlusSquare className="h-4 w-4" style={{ color: COLORS.purple }} />
                    Scan Another Document
                  </button>
                </div>
              </aside>

              {/* Red Flags modal (full list + detector breakdown) */}
              {redFlagsModalOpen && (
                <>
                  <div
                    className="fixed inset-0 z-40 bg-black/40"
                    aria-hidden
                    onClick={() => setRedFlagsModalOpen(false)}
                  />
                  <div
                    className="fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col border-l bg-white shadow-xl"
                    style={{ borderColor: COLORS.border }}
                  >
                    <div className="flex items-center justify-between border-b px-4 py-3" style={{ borderColor: COLORS.border }}>
                      <h3 className="text-[14px] font-semibold" style={{ color: COLORS.textPrimary }}>
                        All Red Flags & Detector Breakdown
                      </h3>
                      <button
                        type="button"
                        onClick={() => setRedFlagsModalOpen(false)}
                        className="rounded p-1 hover:bg-slate-100"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                    <div className="flex-1 overflow-y-auto p-4 space-y-4">
                      <section>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: COLORS.textSecondary }}>
                          Red Flags ({result?.red_flags?.length ?? 0})
                        </h4>
                        <ul className="space-y-2">
                          {(result?.red_flags ?? []).map((flag, i) => (
                            <li key={i} className="flex items-start gap-2 text-[13px]" style={{ color: COLORS.textPrimary }}>
                              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-400" />
                              <span>{flag}</span>
                            </li>
                          ))}
                        </ul>
                      </section>
                      {result?.detector_summary && Object.keys(result.detector_summary).length > 0 && (
                        <section>
                          <h4 className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: COLORS.textSecondary }}>
                            Detector Breakdown
                          </h4>
                          <div className="space-y-3">
                            {Object.entries(result.detector_summary).map(([key, val]) => (
                              <div key={key} className="rounded-lg border p-3" style={{ borderColor: COLORS.border }}>
                                <div className="text-[12px] font-medium capitalize" style={{ color: COLORS.textPrimary }}>
                                  {key.replace(/_/g, " ")}
                                  {val.risk_score != null && (
                                    <span className="ml-2 text-[11px] font-normal" style={{ color: COLORS.textSecondary }}>
                                      (risk: {val.risk_score})
                                    </span>
                                  )}
                                </div>
                                {(val.flags?.length ?? 0) > 0 ? (
                                  <ul className="mt-1.5 list-inside list-disc text-[11px]" style={{ color: COLORS.textSecondary }}>
                                    {val.flags!.map((f, i) => (
                                      <li key={i}>{f}</li>
                                    ))}
                                  </ul>
                                ) : (
                                  <p className="mt-1 text-[11px]" style={{ color: COLORS.textSecondary }}>No flags</p>
                                )}
                              </div>
                            ))}
                          </div>
                        </section>
                      )}
                    </div>
                  </div>
                </>
              )}

              {/* Extracted Text modal */}
              {extractedTextModalOpen && (
                <>
                  <div
                    className="fixed inset-0 z-40 bg-black/40"
                    aria-hidden
                    onClick={() => setExtractedTextModalOpen(false)}
                  />
                  <div
                    className="fixed inset-8 z-50 mx-auto max-w-2xl overflow-hidden rounded-2xl border bg-white shadow-xl flex flex-col"
                    style={{ borderColor: COLORS.border }}
                  >
                    <div className="flex items-center justify-between border-b px-4 py-3" style={{ borderColor: COLORS.border }}>
                      <h3 className="text-[14px] font-semibold" style={{ color: COLORS.textPrimary }}>
                        Extracted Text
                      </h3>
                      <button
                        type="button"
                        onClick={() => setExtractedTextModalOpen(false)}
                        className="rounded p-1 hover:bg-slate-100"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                    <div className="flex-1 overflow-y-auto p-4 text-[13px] leading-relaxed whitespace-pre-wrap" style={{ color: COLORS.textPrimary }}>
                      {result?.extracted_text?.trim() ? result.extracted_text : "No extracted text available for this document."}
                    </div>
                  </div>
                </>
              )}

              {/* Collapsible Forensic Report side-drawer */}
              {forensicDrawerOpen && (
                <>
                  <div
                    className="fixed inset-0 z-30 bg-black/20"
                    aria-hidden
                    onClick={() => setForensicDrawerOpen(false)}
                  />
                  <div
                    className="fixed right-0 top-0 z-40 flex h-full w-full max-w-sm flex-col border-l bg-white shadow-xl"
                    style={{ borderColor: COLORS.border }}
                  >
                    <div className="flex items-center justify-between border-b px-4 py-3" style={{ borderColor: COLORS.border }}>
                      <h3 className="text-[14px] font-semibold" style={{ color: COLORS.textPrimary }}>
                        Forensic Report
                      </h3>
                      <button
                        type="button"
                        onClick={() => setForensicDrawerOpen(false)}
                        className="rounded p-1 hover:bg-slate-100"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                    <div className="flex-1 overflow-y-auto p-4 space-y-5">
                      {/* AI Confidence Details */}
                      <section>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: COLORS.textSecondary }}>
                          AI Confidence Details
                        </h4>
                        <div className="flex items-center gap-4">
                          <div className="relative h-20 w-20 shrink-0">
                            <div className="absolute inset-0 rounded-full bg-slate-200" />
                            <div
                              className="absolute inset-0 rounded-full"
                              style={{
                                backgroundImage: `conic-gradient(#22C55E 0deg ${aiConfidence * 3.6}deg, #E5E7EB 0deg 360deg)`,
                              }}
                            />
                            <div className="absolute inset-3 flex flex-col items-center justify-center rounded-full bg-white">
                              <span className="text-[13px] font-semibold text-slate-900">{aiConfidence}%</span>
                              <span className="text-[10px] font-medium text-slate-500">AI Score</span>
                            </div>
                          </div>
                          <p className="text-[11px]" style={{ color: COLORS.textSecondary }}>
                            How strongly the system believes AI-generated content is present. Higher = stronger AI signals.
                          </p>
                        </div>
                        {(result?.ai_indicators?.length ?? 0) > 0 && (
                          <ul className="mt-2 list-inside list-disc text-[11px]" style={{ color: COLORS.textSecondary }}>
                            {result!.ai_indicators!.map((ind, i) => (
                              <li key={i}>{ind}</li>
                            ))}
                          </ul>
                        )}
                      </section>

                      {/* Extracted Text button */}
                      <section>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: COLORS.textSecondary }}>
                          Extracted Text
                        </h4>
                        <button
                          type="button"
                          onClick={() => setExtractedTextModalOpen(true)}
                          className="flex w-full items-center justify-center gap-2 rounded-lg border px-3 py-2 text-[12px] font-medium hover:bg-slate-50"
                          style={{ borderColor: COLORS.border, color: COLORS.textPrimary }}
                        >
                          <FileText className="h-4 w-4" />
                          View Extracted Text
                        </button>
                      </section>

                      {/* Noise Findings */}
                      <section>
                        <h4 className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: COLORS.textSecondary }}>
                          Noise Findings
                        </h4>
                        <p className="text-[12px] leading-relaxed" style={{ color: COLORS.textPrimary }}>
                          {result?.noise_findings?.trim() ? result.noise_findings : "No noise analysis summary available."}
                        </p>
                      </section>

                      {/* File DNA in drawer */}
                      {result?.file_dna?.length ? (
                        <section>
                          <h4 className="text-[11px] font-semibold uppercase tracking-wider mb-2 flex items-center gap-1.5" style={{ color: COLORS.textSecondary }}>
                            <Dna className="h-3.5 w-3.5" style={{ color: COLORS.purple }} />
                            File DNA
                          </h4>
                          <div className="rounded-lg border overflow-hidden" style={{ borderColor: COLORS.border }}>
                            <table className="w-full text-left text-[11px]">
                              <tbody style={{ color: COLORS.textPrimary }}>
                                {result.file_dna.map((row, i) => (
                                  <tr key={i} className="border-b last:border-b-0" style={{ borderColor: COLORS.border }}>
                                    <td className="px-2 py-1.5 font-medium" style={{ color: COLORS.textSecondary }}>{row.property}</td>
                                    <td className="px-2 py-1.5 break-words">{row.value}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </section>
                      ) : null}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Profile drawer - available from sidebar on any view */}
        {profileOpen && (
          <>
            <div
              className="fixed inset-0 z-40 bg-black/40"
              aria-hidden
              onClick={() => setProfileOpen(false)}
            />
            <div
              className="fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col border-l bg-white shadow-xl"
              style={{ borderColor: COLORS.border }}
            >
              <div className="flex items-center justify-between border-b px-4 py-3" style={{ borderColor: COLORS.border }}>
                <h3 className="text-[14px] font-semibold" style={{ color: COLORS.textPrimary }}>
                  Profile
                </h3>
                <button
                  type="button"
                  onClick={() => setProfileOpen(false)}
                  className="rounded p-1 hover:bg-slate-100"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-5">
                <section className="flex flex-col items-center pt-2">
                  <div
                    className="flex h-16 w-16 items-center justify-center rounded-full text-[20px] font-semibold"
                    style={{ backgroundColor: COLORS.purpleLight, color: COLORS.purple }}
                  >
                    {profileInitials}
                  </div>
                  <p className="mt-2 text-[11px]" style={{ color: COLORS.textSecondary }}>
                    Initials from your display name
                  </p>
                </section>
                <section>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: COLORS.textSecondary }}>
                    Display name
                  </h4>
                  <input
                    type="text"
                    value={userProfile.displayName}
                    onChange={(e) => setUserProfile((p) => ({ ...p, displayName: e.target.value }))}
                    placeholder="Your name"
                    className="w-full rounded-md border bg-[#F1F3F5] px-3 py-2 text-[13px] outline-none focus:ring-1"
                    style={{ borderColor: COLORS.borderDark, color: COLORS.textPrimary }}
                  />
                </section>
                <section>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider mb-2" style={{ color: COLORS.textSecondary }}>
                    Role
                  </h4>
                  <input
                    type="text"
                    value={userProfile.role}
                    onChange={(e) => setUserProfile((p) => ({ ...p, role: e.target.value }))}
                    placeholder="e.g. Senior Ops Analyst"
                    className="w-full rounded-md border bg-[#F1F3F5] px-3 py-2 text-[13px] outline-none focus:ring-1"
                    style={{ borderColor: COLORS.borderDark, color: COLORS.textPrimary }}
                  />
                </section>
                <section>
                  <p className="text-[11px]" style={{ color: COLORS.textSecondary }}>
                    Changes are saved automatically and reflected in the sidebar.
                  </p>
                </section>
              </div>
            </div>
          </>
        )}
      </main>
      {isLoading && (
        <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-white/80 backdrop-blur-md transition-all duration-300">
          <div className="loadingspinner mb-8">
            <div id="square1"></div>
            <div id="square2"></div>
            <div id="square3"></div>
            <div id="square4"></div>
            <div id="square5"></div>
          </div>
          <div className="text-lg font-bold text-gray-800 animate-pulse mt-6 tracking-wide">
            {loadingText}
          </div>
        </div>
      )}
    </div>
  );
}
