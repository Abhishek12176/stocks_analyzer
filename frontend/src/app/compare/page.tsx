"use client";

import { Input } from "@/components/ui/Input";

export default function ComparePage() {
  return (
    <div className="animate-fade-in">
      <div className="border-b border-neutral-800 px-8 py-6">
        <h1 className="text-2xl font-semibold text-neutral-100">Compare</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Coming Soon &mdash; This Feature
        </p>
      </div>

      <div className="p-8 space-y-8">
        <div className="flex items-end gap-3 flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <Input
              placeholder="Coming Soon &mdash; This Feature"
              disabled
            />
          </div>
        </div>

        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-neutral-800/50">
            <svg
              className="h-8 w-8 text-neutral-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold text-neutral-200">
            Coming Soon
          </h2>
          <p className="mt-2 max-w-md text-sm text-neutral-500">
            The compare feature is under development. You&apos;ll be able to
            compare 2–5 stocks side-by-side with radar charts and metric
            tables soon.
          </p>
        </div>
      </div>
    </div>
  );
}
