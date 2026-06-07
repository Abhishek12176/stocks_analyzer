"use client";

import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { cn } from "@/lib/cn";
import { useHistoryStore } from "@/store/historyStore";

export default function HistoryPage() {
  const { items, clear } = useHistoryStore();

  return (
    <div className="animate-fade-in">
      <div className="border-b border-neutral-800 px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-neutral-100">
              History
            </h1>
            <p className="mt-1 text-sm text-neutral-500">
              {items.length} stock{items.length !== 1 ? "s" : ""} analyzed
            </p>
          </div>
          {items.length > 0 ? (
            <Button variant="ghost" size="sm" onClick={clear}>
              Clear history
            </Button>
          ) : null}
        </div>
      </div>

      <div className="p-8">
        {items.length > 0 ? (
          <div className="rounded-xl border border-neutral-800 overflow-hidden">
            {items.map((item, i) => (
              <Link
                key={`${item.symbol}-${i}`}
                href={`/stock/${item.symbol}`}
                className={cn(
                  "flex items-center justify-between px-5 py-4 transition-colors hover:bg-neutral-800/30",
                  i < items.length - 1 && "border-b border-neutral-800/50"
                )}
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm font-semibold text-neutral-200">
                    {item.symbol}
                  </span>
                  <span className="text-sm text-neutral-500">{item.name}</span>
                </div>
                <div className="flex items-center gap-4 text-xs text-neutral-600">
                  {item.signal ? <span>{item.signal}</span> : null}
                  <span>
                    {new Date(item.visitedAt).toLocaleDateString("en-IN", {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No analysis history"
            description="Stocks you analyze will appear here for quick access."
            action={
              <Link href="/stock/RELIANCE">
                <Button variant="primary">Analyze a stock</Button>
              </Link>
            }
          />
        )}
      </div>
    </div>
  );
}
