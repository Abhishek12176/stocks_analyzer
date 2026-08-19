"use client";

import { cn } from "@/lib/cn";

interface MetricCardProps {
  label: string;
  value: string | number;
  change?: {
    value: number;
    direction: "up" | "down" | "flat";
  };
  statusLabel?: string;
  statusColor?: string;
  className?: string;
  loading?: boolean;
  badge?: React.ReactNode;
  trend?: "up" | "down" | "neutral";
}

export function MetricCard({
  label,
  value,
  change,
  statusLabel,
  statusColor,
  className,
  loading = false,
  badge,
  trend,
}: MetricCardProps) {
  if (loading) {
    return (
      <div className={cn("rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-5", className)}>
        <div className="shimmer h-3 w-20 rounded" />
        <div className="shimmer mt-2 h-7 w-28 rounded" />
        <div className="shimmer mt-1.5 h-3 w-16 rounded" />
      </div>
    );
  }

  const isOverbought = statusLabel === "Overbought";
  const isOversold = statusLabel === "Oversold";
  const dotColor = isOverbought ? "var(--color-signal-neutral)" : isOversold ? "var(--color-signal-bullish)" : statusColor || "var(--color-accent-500)";

  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm",
        "p-5 transition-all duration-300 hover:border-neutral-700/60 hover:bg-neutral-900/70",
        "hover:shadow-[0_8px_30px_rgba(0,0,0,0.3)]",
        className
      )}
    >
      <div className="flex items-start justify-between mb-2.5">
        <span className="text-xs font-medium text-neutral-400">
          {label}
        </span>
        {badge && <div className="scale-90">{badge}</div>}
        {statusLabel && !badge && (
          <span className="flex items-center gap-1.5 text-xs font-semibold text-neutral-300">
            <span
              className="inline-block size-1.5 rounded-full"
              style={{ backgroundColor: dotColor }}
            />
            {statusLabel}
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-2.5">
        <span className="font-mono text-[22px] font-bold tracking-tight text-neutral-50">
          {value}
        </span>
        {trend && (
          <span
            className={cn(
              "text-sm font-medium",
              trend === "up" && "text-signal-bullish",
              trend === "down" && "text-signal-bearish",
              trend === "neutral" && "text-neutral-500"
            )}
          >
            {trend === "up" ? "↑" : trend === "down" ? "↓" : "→"}
          </span>
        )}
      </div>
      {change && (
        <div className="mt-1.5">
          <span
            className={cn(
              "inline-flex items-center gap-1 text-xs font-medium",
              change.direction === "up" && "text-signal-bullish",
              change.direction === "down" && "text-signal-bearish",
              change.direction === "flat" && "text-neutral-500"
            )}
          >
            <svg className="size-3" viewBox="0 0 24 24" fill="currentColor">
              {change.direction === "up" && <path d="M12 5l7 7h-5v7h-4v-7H5l7-7z" />}
              {change.direction === "down" && <path d="M12 19l-7-7h5V5h4v7h5l-7 7z" />}
              {change.direction === "flat" && <path d="M5 12h14v2H5z" />}
            </svg>
            {change.value >= 0 ? "+" : ""}{change.value.toFixed(2)}
          </span>
        </div>
      )}
    </div>
  );
}
