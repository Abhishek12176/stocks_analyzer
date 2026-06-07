"use client";

import { useMemo } from "react";

interface SparklineChartProps {
  data: number[];
  color?: string;
  width?: number;
  height?: number;
  smooth?: boolean;
}

function normalize(data: number[], width: number, height: number, padding: number) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  return data.map((v, i) => ({
    x: padding + (i / (data.length - 1)) * plotWidth,
    y: padding + plotHeight - ((v - min) / range) * plotHeight,
  }));
}

function buildPath(points: { x: number; y: number }[], smooth: boolean): string {
  if (points.length < 2) return "";
  if (!smooth || points.length < 3) {
    return points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  }

  const pts = points;
  let d = `M${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const cp1x = pts[i].x + (pts[i + 1].x - pts[i].x) / 3;
    const cp1y = pts[i].y;
    const cp2x = pts[i + 1].x - (pts[i + 1].x - pts[i].x) / 3;
    const cp2y = pts[i + 1].y;
    d += ` C${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${pts[i + 1].x.toFixed(1)},${pts[i + 1].y.toFixed(1)}`;
  }
  return d;
}

export function SparklineChart({
  data,
  color = "#00C853",
  width = 80,
  height = 28,
  smooth = true,
}: SparklineChartProps) {
  const path = useMemo(() => {
    if (!data || data.length < 2) return "";
    const padding = 1;
    const points = normalize(data, width, height, padding);
    return buildPath(points, smooth);
  }, [data, width, height, smooth]);

  if (!data || data.length < 2) {
    return (
      <div
        className="shimmer rounded"
        style={{ width, height }}
      />
    );
  }

  const gradientId = `spark-gradient-${color.replace("#", "")}-${Math.random().toString(36).slice(2, 8)}`;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="flex-shrink-0"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={color} stopOpacity="0.15" />
          <stop offset="100%" stopColor={color} stopOpacity="0.6" />
        </linearGradient>
      </defs>
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="transition-colors duration-300"
      />
    </svg>
  );
}
