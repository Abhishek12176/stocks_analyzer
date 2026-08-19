"use client";

import { forwardRef } from "react";
import { cn } from "@/lib/cn";

const variants = {
  primary:
    "bg-accent-500 text-white hover:bg-accent-600 active:bg-accent-700",
  secondary:
    "bg-neutral-800 text-neutral-100 hover:bg-neutral-700 active:bg-neutral-600 border border-neutral-700",
  ghost:
    "text-neutral-400 hover:text-neutral-100 hover:bg-neutral-800 active:bg-neutral-700",
  danger:
    "bg-signal-bearish/10 text-signal-bearish hover:bg-signal-bearish/20 active:bg-signal-bearish/30 border border-signal-bearish/20",
  signal:
    "",
} as const;

const sizes = {
  sm: "h-8 px-3 text-xs",
  md: "h-10 px-4 text-sm",
  lg: "h-12 px-6 text-base",
} as const;

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: keyof typeof variants;
  size?: keyof typeof sizes;
  loading?: boolean;
  fullWidth?: boolean;
  signal?: "bullish" | "bearish" | "neutral";
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = "primary",
      size = "md",
      loading = false,
      fullWidth = false,
      signal,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const signalStyles = signal
      ? {
          bullish:
            "bg-signal-bullish-bg text-signal-bullish hover:bg-signal-bullish/20 border border-signal-bullish/20",
          bearish:
            "bg-signal-bearish-bg text-signal-bearish hover:bg-signal-bearish/20 border border-signal-bearish/20",
          neutral:
            "bg-signal-neutral-bg text-signal-neutral hover:bg-signal-neutral/20 border border-signal-neutral/20",
        }[signal]
      : "";

    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-all duration-150",
          "active:scale-[0.97]",
          "disabled:opacity-40 disabled:pointer-events-none",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/50",
          variant !== "signal" && variants[variant],
          signal && signalStyles,
          sizes[size],
          fullWidth && "w-full",
          className
        )}
        {...props}
      >
        {loading ? (
          <span className="inline-block size-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
        ) : null}
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";
