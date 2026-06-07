"use client";

import { useEffect, useRef, useMemo } from "react";
import { createChart, ColorType, type IChartApi, type ISeriesApi, type LineSeriesPartialOptions } from "lightweight-charts";
import { calculateRSI } from "@/lib/indicators";
import { cn } from "@/lib/cn";
import type { StockPrice } from "@/types/stock";

interface RsiChartProps {
  history?: StockPrice[] | null;
  loading?: boolean;
  height?: number;
}

export function RsiChart({ history, loading, height = 180 }: RsiChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const overboughtRef = useRef<ISeriesApi<"Line"> | null>(null);
  const oversoldRef = useRef<ISeriesApi<"Line"> | null>(null);
  const middleRef = useRef<ISeriesApi<"Line"> | null>(null);

  const rsiData = useMemo(() => {
    if (!history || history.length < 15) return [];
    const closes = history.map((d) => d.close);
    const dates = history.map((d) => d.date);
    const values = calculateRSI(closes, 14);
    return values.map((v, i) => ({
      time: dates[i + 14]?.split("T")[0] ?? "",
      value: Math.round(v.value * 10) / 10,
    }));
  }, [history]);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#64708A",
        fontSize: 11,
        fontFamily: "JetBrains Mono, SF Mono, Consolas, monospace",
      },
      grid: {
        vertLines: { color: "transparent" },
        horzLines: { color: "#1E2937" },
      },
      timeScale: {
        borderColor: "transparent",
        visible: true,
        tickMarkFormatter: (time: number) => {
          const date = new Date(time * 1000);
          return date.toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
        },
      },
      rightPriceScale: {
        borderColor: "transparent",
        scaleMargins: { top: 0.08, bottom: 0.08 },
        minimumWidth: 40,
      },
      width: containerRef.current.clientWidth,
      height,
      handleScroll: false,
      handleScale: false,
    });

    chartRef.current = chart;

    const obLine = chart.addLineSeries({
      color: "rgba(255, 82, 82, 0.4)",
      lineWidth: 1,
      lineStyle: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    } as LineSeriesPartialOptions);
    overboughtRef.current = obLine;
    obLine.setData(rsiData.map((d) => ({ time: d.time, value: 70 })));

    const osLine = chart.addLineSeries({
      color: "rgba(0, 200, 83, 0.4)",
      lineWidth: 1,
      lineStyle: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    } as LineSeriesPartialOptions);
    oversoldRef.current = osLine;
    osLine.setData(rsiData.map((d) => ({ time: d.time, value: 30 })));

    const midLine = chart.addLineSeries({
      color: "rgba(148, 163, 184, 0.2)",
      lineWidth: 1,
      lineStyle: 3,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    } as LineSeriesPartialOptions);
    middleRef.current = midLine;
    midLine.setData(rsiData.map((d) => ({ time: d.time, value: 50 })));

    const lineSeries = chart.addLineSeries({
      color: "#8B5CF6",
      lineWidth: 3 as const,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: "#8B5CF6",
      crosshairMarkerBackgroundColor: "#8B5CF6",
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
  }, [height, rsiData]);

  useEffect(() => {
    if (!seriesRef.current || rsiData.length === 0) return;
    seriesRef.current.setData(rsiData);
  }, [rsiData]);

  if (loading) {
    return <div className="shimmer rounded-xl" style={{ height }} />;
  }

  if (!history || history.length < 15) {
    return (
      <div className="flex items-center justify-center rounded-xl border border-dashed border-neutral-700 bg-neutral-900/30" style={{ height }}>
        <p className="text-xs text-neutral-500">Insufficient data for RSI</p>
      </div>
    );
  }

  return (
    <div className="relative">
      <div ref={containerRef} />
      <div className="absolute top-2 left-3 flex items-center gap-3 pointer-events-none">
        <span className="flex items-center gap-1.5 text-[10px] text-neutral-500">
          <span className="size-2 rounded-full bg-signal-bearish/60" /> Overbought 70
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-neutral-500">
          <span className="size-2 rounded-full bg-signal-bullish/60" /> Oversold 30
        </span>
      </div>
    </div>
  );
}
