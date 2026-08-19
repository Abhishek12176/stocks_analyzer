"use client";

import { cn } from "@/lib/cn";

type BadgeVariant = "default" | "bullish" | "bearish" | "neutral" | "accent";

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: "bg-neutral-800/60 text-neutral-300 border-neutral-700/60",
  bullish: "bg-signal-bullish/10 text-signal-bullish border-signal-bullish/20",
  bearish: "bg-signal-bearish/10 text-signal-bearish border-signal-bearish/20",
  neutral: "bg-signal-neutral/10 text-signal-neutral border-signal-neutral/20",
  accent: "bg-accent-500/10 text-accent-400 border-accent-500/20",
};

export function Badge({
  variant = "default",
  children,
  className,
}: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-[0.5px]",
        variantStyles[variant],
        className
      )}
    >
      {children}
    </span>
  );
}
