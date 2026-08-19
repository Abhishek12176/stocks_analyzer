"use client";

import { cn } from "@/lib/cn";

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  options: { value: string; label: string }[];
  placeholder?: string;
  error?: string;
}

export function Select({
  className,
  options,
  placeholder,
  error,
  ...props
}: SelectProps) {
  return (
    <div className="relative w-full">
      <select
        className={cn(
          "w-full rounded-lg border bg-transparent px-4 py-2.5 text-sm text-neutral-100 transition-all duration-150 appearance-none",
          "focus:outline-none focus:ring-2 focus:ring-accent-500/30 focus:border-accent-500/50",
          "disabled:opacity-40 disabled:pointer-events-none",
          error
            ? "border-signal-bearish/50"
            : "border-neutral-700 hover:border-neutral-600",
          className
        )}
        {...props}
      >
        {placeholder ? (
          <option value="" className="bg-neutral-900">
            {placeholder}
          </option>
        ) : null}
        {options.map((opt) => (
          <option key={opt.value} value={opt.value} className="bg-neutral-900">
            {opt.label}
          </option>
        ))}
      </select>
      <svg
        className="absolute right-3 top-1/2 -translate-y-1/2 size-4 text-neutral-500 pointer-events-none"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </div>
  );
}
