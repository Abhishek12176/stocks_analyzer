"use client";

import { cn } from "@/lib/cn";

interface ProgressRingProps {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
  label?: string;
  sublabel?: string;
}

export function ProgressRing({
  value,
  max = 100,
  size = 96,
  strokeWidth = 6,
  className,
  label,
  sublabel,
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(value / max, 1);
  const offset = circumference * (1 - progress);

  const color =
    value >= 80
      ? "var(--color-signal-bullish)"
      : value >= 60
        ? "var(--color-signal-bullish)"
        : value >= 40
          ? "var(--color-signal-neutral)"
          : value >= 20
            ? "var(--color-signal-bearish)"
            : "var(--color-signal-bearish)";

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-neutral-800"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div
        className="absolute flex flex-col items-center justify-center"
        style={{ width: size, height: size }}
      >
        <span className="font-mono text-2xl font-bold text-neutral-100">
          {Math.round(value)}
        </span>
        {label ? (
          <span className="text-xs text-neutral-500">{label}</span>
        ) : null}
      </div>
      {sublabel ? (
        <span className="mt-2 text-sm text-neutral-400">{sublabel}</span>
      ) : null}
    </div>
  );
}
