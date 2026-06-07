"use client";

import { useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/cn";

type ToastVariant = "success" | "error" | "info" | "warning";

interface ToastData {
  id: string;
  message: string;
  variant: ToastVariant;
}

let toastListeners: ((toast: ToastData) => void)[] = [];

export function toast(message: string, variant: ToastVariant = "info") {
  const id = Math.random().toString(36).slice(2);
  const data: ToastData = { id, message, variant };
  toastListeners.forEach((listener) => listener(data));
}

const variantStyles: Record<ToastVariant, string> = {
  success: "bg-signal-bullish/10 border-signal-bullish/30 text-signal-bullish",
  error: "bg-signal-bearish/10 border-signal-bearish/30 text-signal-bearish",
  info: "bg-accent-500/10 border-accent-500/30 text-accent-400",
  warning: "bg-signal-neutral/10 border-signal-neutral/30 text-signal-neutral",
};

export function ToastContainer() {
  const [toasts, setToasts] = useState<ToastData[]>([]);

  const addToast = useCallback((toast: ToastData) => {
    setToasts((prev) => [...prev, toast]);
  }, []);

  useEffect(() => {
    toastListeners.push(addToast);
    return () => {
      toastListeners = toastListeners.filter((l) => l !== addToast);
    };
  }, [addToast]);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onClose={() => removeToast(t.id)} />
      ))}
    </div>
  );
}

function ToastItem({
  toast,
  onClose,
}: {
  toast: ToastData;
  onClose: () => void;
}) {
  useEffect(() => {
    const timer = setTimeout(onClose, 4000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div
      className={cn(
        "animate-slide-up rounded-lg border px-4 py-3 text-sm font-medium backdrop-blur-sm",
        "flex items-center gap-3",
        variantStyles[toast.variant]
      )}
    >
      <span className="flex-1">{toast.message}</span>
      <button
        onClick={onClose}
        className="opacity-60 hover:opacity-100 transition-opacity"
      >
        <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}
