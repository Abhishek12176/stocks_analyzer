"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, type IChartApi, type ISeriesApi, type LineSeriesPartialOptions } from "lightweight-charts";

interface LineChartProps {
  data: { time: string; value: number }[];
  color?: string;
  height?: number;
  loading?: boolean;
}

export function LineChart({ data, color = "#3B82F6", height = 120, loading }: LineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#64708A",
        fontSize: 10,
        fontFamily: "JetBrains Mono, SF Mono, Consolas, monospace",
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: "transparent" },
        horzLines: { color: "#1F2937" },
      },
      timeScale: {
        borderColor: "transparent",
        visible: false,
      },
      rightPriceScale: {
        borderColor: "transparent",
        visible: false,
      },
      width: containerRef.current.clientWidth,
      height,
      handleScroll: false,
      handleScale: false,
    });

    chartRef.current = chart;

    const lineSeries = chart.addLineSeries({
      color,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    } as LineSeriesPartialOptions);
    seriesRef.current = lineSeries;

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
  }, [color, height]);

  useEffect(() => {
    if (!seriesRef.current || !data || data.length === 0) return;
    seriesRef.current.setData(data);
  }, [data]);

  return (
    <div className="relative">
      {loading ? (
        <div className="shimmer rounded" style={{ height }} />
      ) : data.length === 0 ? null : null}
      <div
        ref={containerRef}
        className={loading || data.length === 0 ? "hidden" : ""}
      />
      <img src="/logo.png" alt="" className="pointer-events-none ml-2 size-7 opacity-60" />
    </div>
  );
}
