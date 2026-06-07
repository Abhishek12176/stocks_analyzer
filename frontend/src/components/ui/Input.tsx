"use client";

import { forwardRef } from "react";
import { cn } from "@/lib/cn";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  leadingIcon?: React.ReactNode;
  trailingIcon?: React.ReactNode;
  error?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, leadingIcon, trailingIcon, error, ...props }, ref) => {
    return (
      <div className="relative w-full">
        {leadingIcon ? (
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500 pointer-events-none">
            {leadingIcon}
          </div>
        ) : null}
        <input
          ref={ref}
          className={cn(
            "w-full rounded-lg border bg-transparent px-4 py-2.5 text-sm text-neutral-100 transition-all duration-150",
            "placeholder:text-neutral-500",
            "focus:outline-none focus:ring-2 focus:ring-accent-500/30 focus:border-accent-500/50",
            "disabled:opacity-40 disabled:pointer-events-none",
            leadingIcon && "pl-10",
            trailingIcon && "pr-10",
            error
              ? "border-signal-bearish/50 focus:ring-signal-bearish/30 focus:border-signal-bearish"
              : "border-neutral-700 hover:border-neutral-600",
            className
          )}
          {...props}
        />
        {trailingIcon ? (
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500">
            {trailingIcon}
          </div>
        ) : null}
        {error ? (
          <p className="mt-1 text-xs text-signal-bearish">{error}</p>
        ) : null}
      </div>
    );
  }
);

Input.displayName = "Input";
