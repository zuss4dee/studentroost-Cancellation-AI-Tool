"use client";

import React, { useState, useRef } from "react";
import { X, UploadCloud, Download, Loader2, CheckCircle2, AlertCircle, Sparkles, Image as ImageIcon, FileText, Eye, EyeOff, Info } from "lucide-react";
import { translateForeignIdJson, PlacementItem } from "../lib/translateIdApi";

interface ForeignIdTranslatorModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ForeignIdTranslatorModal({ isOpen, onClose }: ForeignIdTranslatorModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [translatedData, setTranslatedData] = useState<Record<string, any> | null>(null);
  const [annotatedImageBase64, setAnnotatedImageBase64] = useState<string | null>(null);
  const [originalImageBase64, setOriginalImageBase64] = useState<string | null>(null);
  const [pdfBase64, setPdfBase64] = useState<string | null>(null);
  const [placements, setPlacements] = useState<PlacementItem[]>([]);
  const [showOverlay, setShowOverlay] = useState(true);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<string>("Extracting & translating ID...");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
      setTranslatedData(null);
      setAnnotatedImageBase64(null);
      setOriginalImageBase64(null);
      setPdfBase64(null);
      setPlacements([]);
      setSuccessMessage(null);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
      setError(null);
      setTranslatedData(null);
      setAnnotatedImageBase64(null);
      setOriginalImageBase64(null);
      setPdfBase64(null);
      setPlacements([]);
      setSuccessMessage(null);
    }
  };

  const handleTranslateAndDownload = async () => {
    if (!selectedFile) {
      setError("Please select a foreign ID card or passport file first.");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccessMessage(null);
    setLoadingStatus("Connecting to Gemini Vision engine...");

    try {
      const result = await translateForeignIdJson(selectedFile, (status) => setLoadingStatus(status));
      setTranslatedData(result.translated_data);
      if (result.annotated_image_base64) {
        setAnnotatedImageBase64(result.annotated_image_base64);
      }
      if (result.original_image_base64) {
        setOriginalImageBase64(result.original_image_base64);
      }
      if (result.pdf_base64) {
        setPdfBase64(result.pdf_base64);
      }
      if (result.placements) {
        setPlacements(result.placements);
      }

      // Automatically trigger PNG download of two-column mapping canvas
      if (result.annotated_image_base64) {
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = result.annotated_image_base64;
        a.download = `Translated_${selectedFile.name.replace(/\.[^/.]+$/, "")}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }

      setSuccessMessage("Foreign ID translated! Two-column mapped PNG image with English Translation Rail downloaded.");
    } catch (err: any) {
      setError(err.message || "Failed to translate foreign ID document.");
    } finally {
      setLoading(false);
    }
  };

  const triggerImageDownloadAgain = () => {
    if (!annotatedImageBase64) return;
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = annotatedImageBase64;
    a.download = `Translated_${selectedFile?.name.replace(/\.[^/.]+$/, "") || "ID"}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const triggerPdfDownloadOptional = () => {
    if (!pdfBase64) return;
    const byteCharacters = atob(pdfBase64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: "application/pdf" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    a.download = "Translated_ID_Summary.pdf";
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  const resetSelection = () => {
    setSelectedFile(null);
    setTranslatedData(null);
    setAnnotatedImageBase64(null);
    setOriginalImageBase64(null);
    setPdfBase64(null);
    setPlacements([]);
    setError(null);
    setSuccessMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl shadow-2xl border border-gray-100 w-full max-w-4xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Modal Header */}
        <div className="px-6 py-5 bg-gradient-to-r from-indigo-900 via-purple-900 to-indigo-800 text-white flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/10 rounded-lg">
              <Sparkles className="w-6 h-6 text-purple-300" />
            </div>
            <div>
              <h3 className="font-bold text-lg text-white leading-tight">Foreign ID Translator</h3>
              <p className="text-xs text-indigo-200 mt-0.5">
                Numbered Source-to-Translation Mapping System & Two-Column English Translation Rail
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-white/70 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-5 flex-1">
          {error && (
            <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
              <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
              <div className="flex-1">
                <span className="font-semibold block">Translation Failed</span>
                <span className="text-xs text-red-600">{error}</span>
              </div>
            </div>
          )}

          {successMessage && (
            <div className="flex items-start gap-3 p-4 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-800 text-sm">
              <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
              <div className="flex-1">
                <span className="font-semibold block">Success</span>
                <span className="text-xs text-emerald-700">{successMessage}</span>
              </div>
            </div>
          )}

          {/* Upload Dropzone */}
          {!translatedData ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
                isDragOver
                  ? "border-indigo-500 bg-indigo-50/50 scale-[0.99]"
                  : selectedFile
                  ? "border-emerald-400 bg-emerald-50/30"
                  : "border-gray-200 hover:border-indigo-400 hover:bg-gray-50/50"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.pdf"
                onChange={handleFileChange}
                className="hidden"
              />

              <div className="flex flex-col items-center justify-center space-y-3">
                <div className="w-14 h-14 rounded-full bg-indigo-50 flex items-center justify-center text-indigo-600 shadow-sm">
                  {selectedFile ? <ImageIcon className="w-7 h-7 text-emerald-600" /> : <UploadCloud className="w-7 h-7" />}
                </div>

                {selectedFile ? (
                  <div>
                    <p className="font-semibold text-gray-800 text-base">{selectedFile.name}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for Document Overlay Translation
                    </p>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        resetSelection();
                      }}
                      className="mt-2 text-xs text-red-500 hover:underline font-medium"
                    >
                      Remove file
                    </button>
                  </div>
                ) : (
                  <div>
                    <p className="font-semibold text-gray-700 text-sm">
                      Click to upload or drag & drop Foreign ID
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      Numbered source markers + English Translation Rail. Preserves ID photo & document pixel-for-pixel.
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Results View */
            <div className="space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-gray-100">
                <div className="flex items-center gap-3">
                  <h4 className="font-semibold text-gray-800 text-sm flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                    Two-Column Document Translation Preview
                  </h4>
                  {/* Show Overlays Toggle Switch */}
                  <button
                    type="button"
                    onClick={() => setShowOverlay(!showOverlay)}
                    className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold transition-all border ${
                      showOverlay
                        ? "bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100"
                        : "bg-gray-100 text-gray-600 border-gray-200 hover:bg-gray-200"
                    }`}
                  >
                    {showOverlay ? <Eye className="w-3.5 h-3.5 text-indigo-600" /> : <EyeOff className="w-3.5 h-3.5 text-gray-500" />}
                    <span>{showOverlay ? "Translation Rail: ON" : "Original Card Only"}</span>
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={triggerImageDownloadAgain}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold bg-indigo-600 text-white hover:bg-indigo-700 rounded-lg shadow-sm transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download PNG
                  </button>
                  {pdfBase64 && (
                    <button
                      onClick={triggerPdfDownloadOptional}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 rounded-lg transition-colors"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      PDF Report (Optional)
                    </button>
                  )}
                </div>
              </div>

              {/* Primary Image Output Preview (Two-Column Output Canvas) */}
              <div className="relative bg-slate-900 rounded-2xl p-4 flex items-center justify-center overflow-hidden max-h-[420px] shadow-inner border border-slate-800">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={showOverlay && annotatedImageBase64 ? annotatedImageBase64 : (originalImageBase64 || annotatedImageBase64 || "")}
                  alt="Translated Foreign ID Numbered Mapping Canvas"
                  className="max-h-[390px] w-auto object-contain rounded-lg shadow-lg border border-slate-700"
                />

                {/* Floating Tooltip when hovering over a field row */}
                {hoveredIndex !== null && placements[hoveredIndex] && (
                  <div className="absolute top-6 left-6 z-20 bg-slate-950/90 text-white border border-purple-500/50 rounded-xl p-3 shadow-2xl backdrop-blur-md max-w-sm animate-in fade-in duration-150">
                    <div className="flex items-center gap-1.5 text-[11px] font-bold text-purple-300 uppercase tracking-wider mb-1">
                      <Info className="w-3.5 h-3.5" />
                      Marker #{placements[hoveredIndex].index}
                    </div>
                    <p className="text-xs font-bold text-white">{placements[hoveredIndex].text}</p>
                    <p className="text-[11px] text-purple-200 mt-1 font-mono">
                      Mapped to source field on original ID card
                    </p>
                  </div>
                )}
              </div>

              {/* Extracted JSON Data & Numbered Mapping Table */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <h5 className="font-semibold text-gray-700 text-xs uppercase tracking-wider">
                    Extracted English Fields & Numbered Source Mappings
                  </h5>
                  <span className="text-[11px] text-purple-600 font-medium">Hover row to highlight marker #</span>
                </div>
                <div className="bg-gray-50 rounded-xl border border-gray-100 overflow-hidden max-h-48 overflow-y-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-indigo-900 text-white font-semibold">
                      <tr>
                        <th className="px-4 py-2 text-center w-12">#</th>
                        <th className="px-4 py-2">Field Name</th>
                        <th className="px-4 py-2">English Translation</th>
                        <th className="px-4 py-2 text-right">Mapping Marker</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200/60">
                      {Object.entries(translatedData).map(([key, val], idx) => {
                        const placement = placements[idx];
                        const isHovered = hoveredIndex === idx;
                        return (
                          <tr
                            key={idx}
                            onMouseEnter={() => setHoveredIndex(idx)}
                            onMouseLeave={() => setHoveredIndex(null)}
                            className={`transition-colors ${
                              isHovered
                                ? "bg-purple-100/80 font-medium text-purple-950"
                                : idx % 2 === 0
                                ? "bg-white"
                                : "bg-gray-50/50"
                            }`}
                          >
                            <td className="px-4 py-2 font-bold text-center border-r border-gray-100">
                              <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-purple-900 text-white text-[11px]">
                                {idx + 1}
                              </span>
                            </td>
                            <td className="px-4 py-2 font-semibold text-gray-800 border-r border-gray-100">{key}</td>
                            <td className="px-4 py-2 text-gray-700">
                              {typeof val === "object" ? JSON.stringify(val) : String(val)}
                            </td>
                            <td className="px-4 py-2 text-right">
                              <span className="inline-block px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-purple-100 text-purple-900 border border-purple-200">
                                MARKER #{idx + 1}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="flex justify-end">
                <button
                  onClick={resetSelection}
                  className="text-xs text-indigo-600 hover:underline font-medium"
                >
                  Translate another document
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-100 flex items-center justify-between">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-gray-600 hover:bg-gray-200/60 rounded-xl transition-colors"
          >
            Close
          </button>

          {!translatedData && (
            <button
              onClick={handleTranslateAndDownload}
              disabled={!selectedFile || loading}
              className={`inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-xs text-white shadow-md transition-all duration-200 ${
                !selectedFile || loading
                  ? "bg-indigo-300 cursor-not-allowed"
                  : "bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 active:scale-[0.98]"
              }`}
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                  <span>{loadingStatus}</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Translate ID Document</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
