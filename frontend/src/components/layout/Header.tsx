"use client";

import { useUiStore } from "@/store/uiStore";
import { cn } from "@/lib/cn";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";

function getMarketStatus() {
  const opts: Intl.DateTimeFormatOptions = {
    timeZone: "Asia/Kolkata",
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  };
  const parts = new Intl.DateTimeFormat("en-US", opts).formatToParts(new Date());
  const p = (t: string) => parts.find((x) => x.type === t)?.value || "";
  const dayName = p("weekday");
  const month = p("month");
  const day = parseInt(p("day"), 10);
  const year = p("year");
  const hour = parseInt(p("hour"), 10);
  const minute = parseInt(p("minute"), 10);

  const isWeekday = !["Sat", "Sun"].includes(dayName);
  const afterOpen = hour > 9 || (hour === 9 && minute >= 0);
  const beforeClose = hour < 15 || (hour === 15 && minute <= 30);

  return {
    open: isWeekday && afterOpen && beforeClose,
    date: `${month} ${day}, ${year}`,
  };
}

export function Header() {
  const openCommandPalette = useUiStore((s) => s.openCommandPalette);
  const [status, setStatus] = useState({ open: false, date: "" });

  useEffect(() => {
    setStatus(getMarketStatus());
    const id = setInterval(() => setStatus(getMarketStatus()), 60000);
    return () => clearInterval(id);
  }, []);

  return (
    <motion.header
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className="flex h-14 items-center justify-between border-b border-neutral-800 bg-[var(--color-bg-sidebar)]/80 backdrop-blur-xl px-6"
    >
      {/* Left - Market Status */}
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "flex items-center gap-2 rounded-lg px-3 py-1.5",
            status.open
              ? "bg-signal-bullish/10"
              : "bg-signal-bearish/10"
          )}
        >
          <span className="relative flex size-2">
            {status.open && (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal-bullish opacity-75" />
            )}
            <span
              className={cn(
                "relative inline-flex size-2 rounded-full",
                status.open ? "bg-signal-bullish" : "bg-signal-bearish"
              )}
            />
          </span>
          <span
            className={cn(
              "text-xs font-semibold",
              status.open ? "text-signal-bullish" : "text-signal-bearish"
            )}
          >
            {status.open ? "MARKET OPEN" : "MARKET CLOSED"}
          </span>
        </div>
        <span className="text-xs text-neutral-400 font-medium">NSE — {status.date}</span>
      </div>

      {/* Right */}
      <div className="flex items-center gap-2">
        {/* Global Search */}
        <button
          onClick={() => openCommandPalette()}
          className={cn(
            "flex items-center gap-2 rounded-xl border border-neutral-800 bg-neutral-900 px-3 py-1.5",
            "text-sm text-neutral-500 hover:text-neutral-50 hover:border-accent-500/30 transition-all duration-200"
          )}
        >
          <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
          </svg>
          <span className="hidden sm:inline">Search stocks...</span>
          <kbd className="ml-1 rounded-md border border-neutral-800 bg-neutral-800/50 px-1.5 py-0.5 text-[11px] text-neutral-500">
            ⌘K
          </kbd>
        </button>

        {/* Notifications */}
        <button className="relative rounded-xl p-2 text-neutral-500 hover:bg-neutral-800/50 hover:text-neutral-50 transition-colors">
          <svg className="size-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9" /><path d="M13.73 21a2 2 0 01-3.46 0" />
          </svg>
          <span className="absolute right-1.5 top-1.5 size-2 rounded-full bg-accent-500" />
        </button>

        {/* Profile */}
        <button className="flex items-center gap-2 rounded-xl p-1.5 text-neutral-500 hover:bg-neutral-800/50 transition-colors">
          <img src="/logo.png" alt="AVORA" className="size-8 rounded-lg animate-logo-glow" />
        </button>
      </div>
    </motion.header>
  );
}