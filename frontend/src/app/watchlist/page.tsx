"use client";

import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { useWatchlistStore } from "@/store/watchlistStore";

export default function WatchlistPage() {
  const { items, remove, clear } = useWatchlistStore();

  return (
    <div className="animate-fade-in">
      <div className="border-b border-neutral-800 px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-neutral-100">
              Watchlist
            </h1>
            <p className="mt-1 text-sm text-neutral-500">
              {items.length} stock{items.length !== 1 ? "s" : ""} tracked
            </p>
          </div>
          {items.length > 0 ? (
            <Button variant="ghost" size="sm" onClick={clear}>
              Clear all
            </Button>
          ) : null}
        </div>
      </div>

      <div className="p-8">
        {items.length > 0 ? (
          <div className="space-y-2">
            {items.map((item) => (
              <div
                key={item.symbol}
                className="flex items-center justify-between rounded-xl border border-neutral-800 bg-neutral-900 px-5 py-4 transition-colors hover:border-neutral-700"
              >
                <Link
                  href={`/stock/${item.symbol}`}
                  className="flex items-center gap-3 flex-1"
                >
                  <span className="font-mono text-base font-semibold text-neutral-200">
                    {item.symbol}
                  </span>
                  <span className="text-sm text-neutral-500">{item.name}</span>
                </Link>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-neutral-600">
                    Added {new Date(item.addedAt).toLocaleDateString("en-IN")}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => remove(item.symbol)}
                  >
                    Remove
                  </Button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState
            title="Your watchlist is empty"
            description="Start tracking stocks by adding them from the stock detail page."
            action={
              <Link href="/">
                <Button variant="primary">Browse stocks</Button>
              </Link>
            }
          />
        )}
      </div>
    </div>
  );
}
