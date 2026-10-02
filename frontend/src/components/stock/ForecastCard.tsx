"use client";

import { useMemo } from "react";
import { cn } from "@/lib/cn";
import { SIGNAL_COLORS } from "@/lib/constants";
import { formatPrice, formatPercent } from "@/lib/formatters";
import { Skeleton } from "@/components/ui/Skeleton";
import { Badge } from "@/components/ui/Badge";
import type { ForecastResponse } from "@/types/forecast";

interface ForecastCardProps {
  forecast?: ForecastResponse | null;
  loading?: boolean;
  retrying?: boolean;
  error?: boolean;
  errorMessage?: string;
  onRetry?: () => void;
}

function toSignalColors(direction?: number) {
  if (direction != null && direction > 0) return SIGNAL_COLORS.bullish;
  if (direction != null && direction < 0) return SIGNAL_COLORS.bearish;
  return SIGNAL_COLORS.neutral;
}

function formatPct0(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(1)}%`;
}

export function ForecastCard({ forecast, loading, retrying, error, errorMessage, onRetry }: ForecastCardProps) {
  const unavailable = useMemo(() => {
    if (!forecast) return false;
    return forecast.isAvailable === false || !!forecast.error;
  }, [forecast]);

  if (loading || retrying) {
    return (
      <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-6 space-y-4">
        <Skeleton className="h-5 w-48" />
        <div className="flex items-center gap-6">
          <Skeleton className="size-[72px] rounded-full" />
          <div className="space-y-2 flex-1">
            <Skeleton className="h-8 w-32" />
            <Skeleton className="h-4 w-56" />
          </div>
        </div>
        <Skeleton className="h-32 rounded-xl" />
        <Skeleton className="h-28 rounded-xl" />
      </div>
    );
  }

  if (!forecast) {
    if (error) {
      return (
        <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm px-6 py-10 flex flex-col items-center text-center">
          <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
            <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-neutral-300">Forecast unavailable</h3>
          <p className="mt-1 text-sm text-neutral-500 max-w-sm">
            {errorMessage || "Unable to compute probabilistic forecast. Please try again."}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-6 rounded-xl bg-accent-500/10 px-5 py-2.5 text-sm font-medium text-accent-500 border border-accent-500/20 hover:bg-accent-500/20 transition-colors"
            >
              Retry
            </button>
          )}
        </div>
      );
    }
    return null;
  }

  if (unavailable) {
    return (
      <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm px-6 py-10 flex flex-col items-center text-center">
        <div className="mb-4 size-12 rounded-xl border border-dashed border-neutral-700 flex items-center justify-center text-neutral-600">
          <svg className="size-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h3 className="text-base font-semibold text-neutral-300">Forecast unavailable</h3>
        <p className="mt-1 text-sm text-neutral-500 max-w-sm">
          {forecast.error || `A probabilistic forecast could not be generated for ${forecast.symbol}.`}
        </p>
      </div>
    );
  }

  const latest = forecast.latest;
  const calibration = forecast.calibration;
  const explanation = forecast.explanation;
  const monitor: NonNullable<ForecastResponse["monitor"]> =
    forecast.monitor ?? { mode: "unknown", freshness: "unknown", alarms: [] };
  const colors = toSignalColors(latest?.direction);
  const probability = latest?.probability ?? 0;
  const probPct = Math.round(probability * 100);
  const calMethod = calibration?.method ?? "uncalibrated";

  const backtestRows = (forecast.backtest ?? []).slice().sort((a, b) => {
    if (a.model === "buy_hold") return 1;
    if (b.model === "buy_hold") return -1;
    return (b.cum_return ?? 0) - (a.cum_return ?? 0);
  });

  const monitorVariant =
    monitor.mode === "fresh"
      ? "bullish"
      : monitor.mode === "degraded"
        ? "bearish"
        : "neutral";

  return (
    <div className="space-y-4">
      <div
        className={cn(
          "relative overflow-hidden rounded-2xl border p-6 transition-all duration-300",
          "hover:shadow-[0_8px_30px_rgba(0,0,0,0.35)]",
          colors.bg,
          colors.border,
        )}
      >
        <div className="absolute top-0 right-0 w-64 h-64 opacity-5 pointer-events-none">
          <div
            className="absolute top-[-60px] right-[-60px] w-48 h-48 rounded-full"
            style={{ background: `radial-gradient(circle, ${colors.text === "text-signal-bullish" ? "var(--color-signal-bullish)" : colors.text === "text-signal-bearish" ? "var(--color-signal-bearish)" : "var(--color-signal-neutral)"}, transparent)` }}
          />
        </div>

        <div className="relative z-10">
          <div className="flex items-center justify-between flex-wrap gap-2 mb-5">
            <div>
              <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[1px]">
                Probabilistic Forecast · {forecast.horizon}d horizon
              </span>
              <div className="flex items-center gap-2 mt-1">
                <span className={cn("text-[26px] font-bold font-mono tracking-tight", colors.text)}>
                  {latest?.signal ?? "HOLD"}
                </span>
                {latest?.direction != null && latest.direction !== 0 ? (
                  <Badge variant={latest.direction > 0 ? "bullish" : "bearish"}>
                    {latest.direction > 0 ? "BULLISH" : "BEARISH"}
                  </Badge>
                ) : (
                  <Badge variant="neutral">NEUTRAL</Badge>
                )}
              </div>
            </div>

            <div className="flex flex-col items-center">
              <div className="relative size-[72px] flex-shrink-0">
                <svg width={72} height={72} viewBox="0 0 72 72" className="transform -rotate-90">
                  <circle cx="36" cy="36" r="31" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="4" />
                  <circle
                    cx="36" cy="36" r="31" fill="none"
                    stroke={colors.text === "text-signal-bullish" ? "var(--color-signal-bullish)" : colors.text === "text-signal-bearish" ? "var(--color-signal-bearish)" : "var(--color-signal-neutral)"}
                    strokeWidth="4" strokeLinecap="round"
                    strokeDasharray={2 * Math.PI * 31}
                    strokeDashoffset={2 * Math.PI * 31 * (1 - probability)}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-base font-bold text-neutral-50">{probPct}%</span>
                  <span className="text-[9px] font-medium text-neutral-500 uppercase tracking-[0.5px]">P(up)</span>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 mb-5">
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3.5 py-2.5">
              <span className="text-[10px] font-medium text-neutral-500 uppercase tracking-[0.5px]">Last Close</span>
              <div className="mt-1 font-mono text-sm font-semibold text-neutral-100">
                {latest?.close != null ? formatPrice(latest.close) : "N/A"}
              </div>
              {latest?.asOf ? (
                <div className="text-[11px] font-mono text-neutral-500">{latest.asOf}</div>
              ) : null}
            </div>
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3.5 py-2.5">
              <span className="text-[10px] font-medium text-neutral-500 uppercase tracking-[0.5px]">Raw P(up)</span>
              <div className="mt-1 font-mono text-sm font-semibold text-neutral-100">
                {formatPct0(latest?.rawProbability)}
              </div>
              <div className="text-[11px] text-neutral-500">before calibration</div>
            </div>
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3.5 py-2.5">
              <span className="text-[10px] font-medium text-neutral-500 uppercase tracking-[0.5px]">Confidence</span>
              <div className="mt-1 font-mono text-sm font-semibold text-neutral-100">
                {latest?.confidence != null ? `${Math.round(latest.confidence * 100)}%` : "N/A"}
              </div>
            </div>
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] px-3.5 py-2.5">
              <span className="text-[10px] font-medium text-neutral-500 uppercase tracking-[0.5px]">Model Status</span>
              <div className="mt-1.5">
                <Badge variant={monitorVariant as "bullish" | "bearish" | "neutral"}>
                  {(monitor.mode ?? "unknown").toUpperCase()}
                </Badge>
              </div>
              {monitor.alarms && monitor.alarms.length > 0 ? (
                <div className="text-[11px] text-signal-bearish mt-1">
                  {monitor.alarms.length} alarm{(monitor.alarms.length > 1 ? "s" : "")}
                </div>
              ) : null}
            </div>
          </div>

          {latest?.reason ? (
            <div className="text-[11px] font-semibold text-neutral-500 uppercase tracking-[0.8px] mb-2">
              Why
            </div>
          ) : null}
          {latest?.reason ? (
            <p className="text-sm text-neutral-300">{latest.reason}</p>
          ) : null}
        </div>
      </div>

      {explanation && explanation.factors.length > 0 ? (
        <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[13px] font-semibold text-neutral-400 uppercase tracking-[1px]">
              Top Factors
            </h3>
            <span className="text-xs font-mono text-neutral-500">{explanation.model}</span>
          </div>
          <div className="space-y-2.5">
            {explanation.factors.map((factor) => {
              const positive = factor.direction === "positive";
              return (
                <div key={factor.factor} className="flex items-center gap-3">
                  <span className="w-44 shrink-0 truncate font-mono text-xs text-neutral-300">
                    {factor.factor}
                  </span>
                  <div className="flex-1 h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                    <div
                      className={cn("h-full rounded-full", positive ? "bg-signal-bullish" : "bg-signal-bearish")}
                      style={{ width: `${Math.min(Math.abs(factor.impact) * 100, 100)}%` }}
                    />
                  </div>
                  <span
                    className={cn(
                      "w-16 shrink-0 text-right font-mono text-xs",
                      positive ? "text-signal-bullish" : "text-signal-bearish"
                    )}
                  >
                    {factor.impact >= 0 ? "+" : ""}
                    {(factor.impact * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-[11px] text-neutral-500">
            Impact = P(up) change for a {explanation.deltaScale.toLowerCase()}
          </p>
        </div>
      ) : null}

      {backtestRows.length > 0 ? (
        <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 backdrop-blur-sm p-6 overflow-x-auto">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[13px] font-semibold text-neutral-400 uppercase tracking-[1px]">
              Out-of-Sample Backtest
            </h3>
            {calibration ? (
              <span className="text-xs font-mono text-neutral-500">
                calib: {calMethod} · n={calibration.n}
                {calibration.brierCalibrated != null
                  ? ` · brier=${calibration.brierCalibrated.toFixed(3)}`
                  : ""}
              </span>
            ) : null}
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[11px] text-neutral-500 uppercase tracking-[0.5px]">
                <th className="text-left font-medium py-2 pr-4">Model</th>
                <th className="text-right font-medium py-2 px-2">P&L</th>
                <th className="text-right font-medium py-2 px-2">Acc</th>
                <th className="text-right font-medium py-2 px-2">Win</th>
                <th className="text-right font-medium py-2 px-2">Max DD</th>
                <th className="text-right font-medium py-2 px-2">Sharpe</th>
                <th className="text-right font-medium py-2 px-2">Trades</th>
              </tr>
            </thead>
            <tbody>
              {backtestRows.map((row) => (
                <tr key={row.model} className="border-t border-neutral-800/60">
                  <td className="py-2 pr-4 font-mono text-xs text-neutral-300">{row.model}</td>
                  <td
                    className={cn(
                      "py-2 px-2 text-right font-mono text-xs",
                      (row.cum_return ?? 0) >= 0 ? "text-signal-bullish" : "text-signal-bearish"
                    )}
                  >
                    {formatPercent(row.cum_return != null ? row.cum_return * 100 : null)}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-xs text-neutral-300">
                    {row.accuracy != null ? `${(row.accuracy * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-xs text-neutral-300">
                    {row.win_rate != null ? `${(row.win_rate * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-xs text-neutral-300">
                    {row.max_dd != null ? `${(row.max_dd * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-xs text-neutral-300">
                    {row.sharpe != null ? row.sharpe.toFixed(2) : "—"}
                  </td>
                  <td className="py-2 px-2 text-right font-mono text-xs text-neutral-300">
                    {row.n_trades != null ? row.n_trades : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div className="flex items-center justify-between flex-wrap gap-2 text-[11px] text-neutral-500">
        <span className="font-mono">
          generated at {new Date(forecast.generatedAt).toLocaleString("en-IN", {
            day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit",
          })}
        </span>
        {forecast.versions ? (
          <span className="font-mono">
            feat {forecast.versions.feature ?? "?"} · model {forecast.versions.model ?? "?"} · ens {forecast.versions.ensemble ?? "?"}
          </span>
        ) : null}
      </div>
    </div>
  );
}