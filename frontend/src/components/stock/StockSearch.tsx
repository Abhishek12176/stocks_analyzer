"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import { cn } from "@/lib/cn";
import { useDebounce } from "@/hooks/useDebounce";
import type { SearchResponse, StockSymbol } from "@/types/api";

export function StockSearch() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const debouncedQuery = useDebounce(query, 300);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const { data, isLoading } = useQuery<SearchResponse>({
    queryKey: ["stockSearch", debouncedQuery],
    queryFn: () => apiGet<SearchResponse>("/stock/search", { q: debouncedQuery }),
    enabled: debouncedQuery.length >= 1,
    staleTime: 60_000,
  });

  const results = data?.results ?? [];

  const handleSelect = useCallback(
    (symbol: StockSymbol) => {
      setQuery("");
      setIsOpen(false);
      router.push(`/stock/${symbol.symbol}`);
    },
    [router]
  );

  useEffect(() => {
    setActiveIndex(-1);
  }, [results]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        !inputRef.current?.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) return;

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setActiveIndex((prev) => Math.min(prev + 1, results.length - 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setActiveIndex((prev) => Math.max(prev - 1, 0));
        break;
      case "Enter":
        e.preventDefault();
        if (activeIndex >= 0 && activeIndex < results.length) {
          handleSelect(results[activeIndex]);
        }
        break;
      case "Escape":
        setIsOpen(false);
        inputRef.current?.blur();
        break;
    }
  };

  return (
    <div className="relative w-full">
      <div className="relative group">
        <svg
          className="absolute left-3.5 top-1/2 -translate-y-1/2 size-4 text-neutral-500 group-focus-within:text-accent-400 transition-colors"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => {
            if (query.length >= 1) setIsOpen(true);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Search stocks by symbol or name..."
          className="w-full h-11 pl-10 pr-10 text-sm bg-neutral-900/80 border border-neutral-800 rounded-xl text-[var(--color-text-primary)] placeholder-neutral-500 focus:outline-none focus:border-accent-500/40 focus:ring-1 focus:ring-accent-500/20 transition-all backdrop-blur-sm"
        />
        {query && (
          <button
            onClick={() => {
              setQuery("");
              setIsOpen(false);
              inputRef.current?.focus();
            }}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500 hover:text-neutral-300 transition-colors"
          >
            <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {isOpen && (query.length >= 1) && (
        <div
          ref={dropdownRef}
          className="absolute top-full mt-1.5 w-full rounded-xl border border-neutral-800 bg-[var(--color-bg)] shadow-2xl z-50 overflow-hidden backdrop-blur-xl"
        >
          {isLoading ? (
            <div className="p-4 text-sm text-neutral-500 text-center flex items-center justify-center gap-2">
              <span className="inline-block size-3.5 border-2 border-neutral-600 border-t-neutral-300 rounded-full animate-spin" />
              Searching...
            </div>
          ) : results.length === 0 ? (
            <div className="p-5 text-sm text-neutral-500 text-center">
              No results found for &quot;{query}&quot;
            </div>
          ) : (
            <ul className="py-1.5 max-h-80 overflow-y-auto">
              {results.map((result, i) => (
                <li
                  key={`${result.symbol}-${result.exchange}`}
                  onClick={() => handleSelect(result)}
                  className={cn(
                    "flex items-center gap-3 px-4 py-3 cursor-pointer transition-all duration-150",
                    i === activeIndex
                      ? "bg-accent-500/10 text-neutral-50"
                      : "text-neutral-300 hover:bg-white/[0.03]"
                  )}
                >
                  <span className="font-mono text-sm font-bold text-neutral-50">
                    {result.symbol}
                  </span>
                  <span className="text-sm text-neutral-500 truncate flex-1">
                    {result.name}
                  </span>
                  <span className="text-[10px] font-semibold text-neutral-600 uppercase tracking-[0.5px] px-1.5 py-0.5 rounded border border-neutral-800">
                    {result.exchange}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
