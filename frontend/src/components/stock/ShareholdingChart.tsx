"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  type TooltipProps,
} from "recharts";
import { Skeleton } from "@/components/ui/Skeleton";
import type { QuarterlyHolding } from "@/types/shareholding";

interface ShareholdingChartProps {
  quarterlyData?: QuarterlyHolding[] | null;
  loading?: boolean;
}

  const CATEGORIES = [
    { key: "promoter", label: "Promoter", color: "var(--color-signal-bullish)" },
    { key: "fii", label: "FII/FPI", color: "var(--color-accent-500)" },
    { key: "dii", label: "DII", color: "var(--color-signal-neutral)" },
    { key: "government", label: "Government", color: "var(--color-signal-bearish)" },
    { key: "public", label: "Public", color: "var(--color-text-muted)" },
  ] as const;

function toPercent(val: string): number {
  const n = parseFloat(val);
  return isNaN(n) ? 0 : n;
}

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="rounded-xl border border-neutral-700/50 bg-neutral-900/95 backdrop-blur-sm px-4 py-3 shadow-xl">
      <p className="text-sm font-semibold text-neutral-200 mb-2">{label}</p>
      <div className="space-y-1.5">
        {payload.map((entry) => (
          <div key={entry.dataKey} className="flex items-center justify-between gap-6">
            <div className="flex items-center gap-2">
              <div
                className="size-2.5 rounded-sm"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-xs text-neutral-400">{entry.name}</span>
            </div>
            <span className="font-mono text-xs font-semibold text-neutral-100">
              {typeof entry.value === "number" ? entry.value.toFixed(2) : "0.00"}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CustomLegend({ payload }: { payload?: { value: string; color: string }[] }) {
  if (!payload) return null;
  return (
    <div className="flex flex-wrap justify-center gap-5 pt-2">
      {payload.map((entry) => (
        <div key={entry.value} className="flex items-center gap-2">
          <div
            className="size-3 rounded-sm"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-xs font-medium text-neutral-400">
            {entry.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function ShareholdingChart({ quarterlyData, loading }: ShareholdingChartProps) {
  const chartData = useMemo(() => {
    if (!quarterlyData) return [];
    return quarterlyData.map((q) => ({
      quarter: q.quarter,
      Promoter: toPercent(q.promoter),
      "FII/FPI": toPercent(q.fii),
      DII: toPercent(q.dii),
      Government: toPercent(q.government),
      Public: toPercent(q.public),
    }));
  }, [quarterlyData]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-5 w-56" />
        <Skeleton className="h-[340px] rounded-xl" />
      </div>
    );
  }

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 rounded-xl border border-dashed border-neutral-700/60 bg-neutral-900/30">
        <p className="text-sm text-neutral-500">No shareholding data available</p>
      </div>
    );
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={340}>
        <BarChart
          data={chartData}
          margin={{ top: 8, right: 12, left: 0, bottom: 4 }}
          barCategoryGap="20%"
          barGap={0}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--color-border)"
            vertical={false}
          />
          <XAxis
            dataKey="quarter"
            tick={{ fontSize: 11, fill: "var(--color-text-muted)", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={{ stroke: "var(--color-border)" }}
            dy={6}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 11, fill: "var(--color-text-muted)", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) => `${v}%`}
            width={40}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
          <Legend content={<CustomLegend />} />
          {CATEGORIES.map((cat) => (
            <Bar
              key={cat.key}
              dataKey={cat.label}
              stackId="shareholding"
              fill={cat.color}
              stroke="none"
              radius={[2, 2, 0, 0]}
              maxBarSize={48}
              isAnimationActive={true}
              animationDuration={600}
              animationEasing="ease-out"
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
