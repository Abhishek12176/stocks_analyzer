"use client";

import { useState } from "react";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/cn";
import { formatDate } from "@/lib/formatters";
import type { NewsArticle } from "@/types/news";

interface NewsFeedProps {
  articles?: NewsArticle[] | null;
  loading?: boolean;
}

export function NewsFeed({ articles, loading }: NewsFeedProps) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  if (loading) {
    return (
      <div className="space-y-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="rounded-2xl border border-neutral-800 bg-[rgba(15,23,42,0.6)] p-5">
            <Skeleton className="h-5 w-3/4 mb-2" />
            <Skeleton className="h-4 w-full mb-1" />
            <Skeleton className="h-4 w-1/2" />
          </div>
        ))}
      </div>
    );
  }

  if (!articles || articles.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-sm text-neutral-500">
        No news articles available
      </div>
    );
  }

  const toggleExpand = (index: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  return (
    <div className="space-y-3">
      {articles.map((article, i) => {
        const isExpanded = expanded.has(i);
        return (
          <div
            key={i}
            className={cn(
              "rounded-2xl border border-neutral-800 bg-[rgba(15,23,42,0.6)] backdrop-blur-sm p-5 transition-all duration-200",
              "hover:border-neutral-700/60 hover:shadow-[0_4px_20px_rgba(0,0,0,0.2)] cursor-pointer"
            )}
            onClick={() => toggleExpand(i)}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-semibold text-neutral-200 leading-snug line-clamp-2">
                  {article.title}
                </h4>
                {isExpanded && article.summary && (
                  <p className="mt-2.5 text-sm text-neutral-500 leading-relaxed">
                    {article.summary}
                  </p>
                )}
              </div>
              <svg
                className={cn(
                  "size-4 shrink-0 mt-0.5 text-neutral-600 transition-transform duration-200",
                  isExpanded && "rotate-180"
                )}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <div className="flex items-center gap-3 mt-3.5 text-xs text-neutral-600">
              {article.sentiment && (
                <span
                  className={cn(
                    "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide",
                    article.sentiment.label === "positive" && "bg-signal-bullish/10 text-signal-bullish border border-signal-bullish/20",
                    article.sentiment.label === "negative" && "bg-signal-bearish/10 text-signal-bearish border border-signal-bearish/20",
                    article.sentiment.label === "neutral" && "bg-neutral-800/50 text-neutral-400 border border-neutral-700/30",
                  )}
                >
                  {article.sentiment.label === "positive" && "▲"}
                  {article.sentiment.label === "negative" && "▼"}
                  {article.sentiment.label === "neutral" && "◆"}
                  {article.sentiment.label}
                </span>
              )}
              <span className="font-medium">{article.source}</span>
              <span className="text-neutral-700">·</span>
              <span>{formatDate(article.publishedDt || article.published)}</span>
              {article.link && (
                <>
                  <span className="text-neutral-700">·</span>
                  <a
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="text-accent-500 hover:text-accent-400 transition-colors font-medium"
                  >
                    Read more →
                  </a>
                </>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
