"use client";

import React, { useState, useRef } from "react";
import { X, UploadCloud, Download, Loader2, CheckCircle2, AlertCircle, Sparkles, Image as ImageIcon } from "lucide-react";
import { generateTranslatedIdOverlay, OverlayResponse } from "../lib/translatedOverlayApi";

interface TranslatedIdOverlayModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function TranslatedIdOverlayModal({ isOpen, onClose }: TranslatedIdOverlayModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState<string>("Translating…");
  const [error, setError] = useState<string | null>(null);
  const [overlayResult, setOverlayResult] = useState<OverlayResponse | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
      setOverlayResult(null);
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
      setOverlayResult(null);
      setSuccessMessage(null);
    }
  };

  const handleGenerateOverlay = async () => {
    if (!selectedFile) {
      setError("Please select a document file first.");
      return;
    }

    setLoading(true);
    setError(null);
    setSuccessMessage(null);
    setLoadingStatus("Translating…");

    try {
      const result = await generateTranslatedIdOverlay(selectedFile, (status) => setLoadingStatus(status));
      setOverlayResult(result);
      setSuccessMessage("Translation complete. Review the document, then download if needed.");
    } catch (err: any) {
      setError("Translation could not be completed.");
      console.error("[TRANSLATION_ERROR]", err);
    } finally {
      setLoading(false);
    }
  };

  const downloadAnnotatedImage = () => {
    if (!overlayResult?.annotated_image_base64) return;
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = overlayResult.annotated_image_base64;
    a.download = `Translated_${selectedFile?.name || "document.jpg"}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const resetSelection = () => {
    setSelectedFile(null);
    setOverlayResult(null);
    setError(null);
    setSuccessMessage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl shadow-2xl border border-gray-100 w-full max-w-4xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Modal Header */}
        <div className="px-6 py-5 bg-gradient-to-r from-purple-900 via-indigo-900 to-slate-900 text-white flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/10 rounded-lg">
              <Sparkles className="w-6 h-6 text-purple-300" />
            </div>
            <div>
              <h3 className="font-bold text-lg text-white leading-tight">Document Translator</h3>
              <p className="text-xs text-indigo-200 mt-0.5">
                Translate identity documents into English
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
          {!overlayResult ? (
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
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready to translate
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
                      Click to upload or drag & drop document
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      Upload a clear image or PDF to create an English version for review.
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Results View */
            <div className="space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-gray-100">
                <h4 className="font-semibold text-gray-800 text-sm flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                  Translated Document
                </h4>

                <button
                  onClick={downloadAnnotatedImage}
                  className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold bg-indigo-600 text-white hover:bg-indigo-700 rounded-lg shadow-sm transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download
                </button>
              </div>

              {/* Primary Image Output Preview */}
              <div className="relative bg-slate-900 rounded-2xl p-4 flex items-center justify-center overflow-hidden max-h-[420px] shadow-inner border border-slate-800">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={overlayResult.annotated_image_base64}
                  alt="Translated Document"
                  className="max-h-[390px] w-auto object-contain rounded-lg shadow-lg border border-slate-700"
                />
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

          {!overlayResult && (
            <button
              onClick={handleGenerateOverlay}
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
                  <span>{selectedFile ? "Translate document" : "Translate"}</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
