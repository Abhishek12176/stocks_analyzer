"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, type IChartApi, type ISeriesApi, type AreaSeriesPartialOptions } from "lightweight-charts";

interface AreaChartProps {
  data: { time: string; value: number }[];
  lineColor?: string;
  topColor?: string;
  bottomColor?: string;
  height?: number;
  loading?: boolean;
}

export function AreaChart({ data, lineColor = "#22C55E", topColor = "rgba(34, 197, 94, 0.3)", bottomColor = "rgba(34, 197, 94, 0.01)", height = 200, loading }: AreaChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#64708A",
        fontSize: 11,
        fontFamily: "JetBrains Mono, SF Mono, Consolas, monospace",
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: "#1F2937" },
        horzLines: { color: "#1F2937" },
      },
      timeScale: {
        borderColor: "#1F2937",
        timeVisible: false,
      },
      rightPriceScale: {
        borderColor: "#1F2937",
      },
      width: containerRef.current.clientWidth,
      height,
      handleScroll: false,
      handleScale: false,
    });

    chartRef.current = chart;

    const areaSeries = chart.addAreaSeries({
      lineColor,
      topColor,
      bottomColor,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    } as AreaSeriesPartialOptions);
    seriesRef.current = areaSeries;

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [lineColor, topColor, bottomColor, height]);

  useEffect(() => {
    if (!seriesRef.current || !data || data.length === 0) return;
    seriesRef.current.setData(data);
  }, [data]);

  return (
    <div className="relative">
      {loading ? (
        <div className="shimmer rounded-lg" style={{ height }} />
      ) : data.length === 0 ? (
        <div
          className="flex items-center justify-center rounded-lg border border-dashed border-neutral-700 bg-neutral-900/50"
          style={{ height }}
        >
          <p className="text-xs text-neutral-500">No data</p>
        </div>
      ) : null}
      <div
        ref={containerRef}
        className={loading || data.length === 0 ? "hidden" : ""}
      />
      <img src="/logo.png" alt="" className="pointer-events-none absolute bottom-2 left-2 size-8 opacity-60" />
    </div>
  );
}
