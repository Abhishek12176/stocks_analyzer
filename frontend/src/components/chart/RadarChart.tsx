"use client";

import { cn } from "@/lib/cn";

interface RadarDataPoint {
  label: string;
  values: { symbol: string; value: number }[];
}

interface RadarChartProps {
  data: RadarDataPoint[];
  symbols: string[];
  colors?: string[];
  size?: number;
  className?: string;
}

const DEFAULT_COLORS = ["#3B82F6", "#22C55E", "#F59E0B", "#EF4444", "#8B5CF6"];

export function RadarChart({ data, symbols, colors = DEFAULT_COLORS, size = 300, className }: RadarChartProps) {
  if (data.length === 0 || symbols.length === 0) {
    return (
      <div className="flex items-center justify-center" style={{ width: size, height: size }}>
        <p className="text-sm text-neutral-500">No data</p>
      </div>
    );
  }

  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 40;
  const angleStep = (2 * Math.PI) / data.length;

  const maxValue = Math.max(
    ...data.flatMap((d) => d.values.map((v) => v.value)),
    1
  );

  function getPoint(index: number, value: number) {
    const angle = angleStep * index - Math.PI / 2;
    const r = (value / maxValue) * radius;
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
    };
  }

  function getPolygonPoints(symbolIndex: number) {
    return data
      .map((d, i) => {
        const p = getPoint(i, d.values[symbolIndex]?.value ?? 0);
        return `${p.x},${p.y}`;
      })
      .join(" ");
  }

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <svg width={size} height={size}>
        {Array.from({ length: 5 }).map((_, i) => {
          const r = (radius / 5) * (i + 1);
          const pts = Array.from({ length: data.length })
            .map((_, j) => {
              const angle = angleStep * j - Math.PI / 2;
              return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
            })
            .join(" ");
          return (
            <polygon
              key={i}
              points={pts}
              fill="none"
              stroke="#1F2937"
              strokeWidth={1}
            />
          );
        })}

        {data.map((d, i) => {
          const p = getPoint(i, maxValue);
          return (
            <line
              key={d.label}
              x1={cx}
              y1={cy}
              x2={p.x}
              y2={p.y}
              stroke="#1F2937"
              strokeWidth={1}
            />
          );
        })}

        {symbols.map((_, si) => (
          <polygon
            key={si}
            points={getPolygonPoints(si)}
            fill={colors[si % colors.length]}
            fillOpacity={0.1}
            stroke={colors[si % colors.length]}
            strokeWidth={2}
          />
        ))}

        {data.map((d, i) => {
          const p = getPoint(i, maxValue);
          return (
            <text
              key={d.label}
              x={p.x}
              y={p.y}
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-neutral-400"
              fontSize={10}
              fontFamily="Inter, system-ui, sans-serif"
            >
              {d.label}
            </text>
          );
        })}
      </svg>

      <div className="flex flex-wrap gap-4 mt-4 justify-center">
        {symbols.map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            <div
              className="size-3 rounded-sm"
              style={{ backgroundColor: colors[i % colors.length] }}
            />
            <span className="text-xs text-neutral-400 font-mono">{s}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
