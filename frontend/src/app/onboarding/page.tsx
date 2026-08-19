"use client";

import Link from "next/link";
import { Button } from "@/components/ui/Button";

const STEPS = [
  {
    icon: (
      <svg className="size-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
    title: "Search any stock",
    description: "Type an NSE or BSE symbol to get instant analysis.",
  },
  {
    icon: (
      <svg className="size-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    title: "Multi-factor analysis",
    description: "Technical, fundamental, ownership, and news — all in one place.",
  },
  {
    icon: (
      <svg className="size-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
    title: "Data you can trust",
    description: "All data sourced from yfinance, Screener.in, NSE, and NewsData.io — no fake data.",
  },
  {
    icon: (
      <svg className="size-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
      </svg>
    ),
    title: "Build your watchlist",
    description: "Save stocks to track and revisit anytime.",
  },
];

export default function OnboardingPage() {
  return (
    <div className="min-h-screen flex items-center justify-center p-8">
      <div className="max-w-2xl w-full animate-fade-in">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center size-16 rounded-2xl bg-accent-500/10 mb-6">
            <span className="font-mono text-3xl font-bold text-accent-500">
              A
            </span>
          </div>
          <h1 className="text-3xl font-bold text-neutral-100">
            Welcome to AVORA
          </h1>
          <p className="mt-3 text-neutral-500 max-w-md mx-auto leading-relaxed">
            Your premium Indian stock analysis platform. Get actionable trade
            signals backed by technical, fundamental, and ownership data.
          </p>
        </div>

        <div className="grid gap-4 mb-12">
          {STEPS.map((step, i) => (
            <div
              key={i}
              className="flex items-start gap-5 rounded-xl border border-neutral-800 bg-neutral-900/50 p-5"
            >
              <div className="shrink-0 flex items-center justify-center size-12 rounded-xl bg-accent-500/5 text-accent-400">
                {step.icon}
              </div>
              <div>
                <h3 className="text-sm font-semibold text-neutral-200">
                  {step.title}
                </h3>
                <p className="mt-1 text-sm text-neutral-500">
                  {step.description}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="text-center">
          <Link href="/stock/RELIANCE">
            <Button variant="primary" size="lg">
              Start analyzing
            </Button>
          </Link>
          <p className="mt-3 text-xs text-neutral-600">
            Try RELIANCE, TCS, HDFCBANK, or any NSE/BSE symbol
          </p>
        </div>
      </div>
    </div>
  );
}
