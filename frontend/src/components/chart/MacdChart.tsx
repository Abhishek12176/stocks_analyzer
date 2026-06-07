"use client";

import { useEffect, useRef, useMemo } from "react";
import { createChart, ColorType, type IChartApi, type ISeriesApi, type LineSeriesPartialOptions, type HistogramSeriesPartialOptions } from "lightweight-charts";
import { calculateMACD } from "@/lib/indicators";
import type { StockPrice } from "@/types/stock";

interface MacdChartProps {
  history?: StockPrice[] | null;
  loading?: boolean;
  height?: number;
}

export function MacdChart({ history, loading, height = 180 }: MacdChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const macdRef = useRef<ISeriesApi<"Line"> | null>(null);
  const signalRef = useRef<ISeriesApi<"Line"> | null>(null);
  const histRef = useRef<ISeriesApi<"Histogram"> | null>(null);

  const macdData = useMemo(() => {
    if (!history || history.length < 35) return null;
    const closes = history.map((d) => d.close);
    const dates = history.map((d) => d.date);
    return calculateMACD(closes, dates);
  }, [history]);

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
        scaleMargins: { top: 0.12, bottom: 0.12 },
      },
      width: containerRef.current.clientWidth,
      height,
      handleScroll: false,
      handleScale: false,
    });

    chartRef.current = chart;

    const macdSeries = chart.addLineSeries({
      color: "#3B82F6",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: "#3B82F6",
      crosshairMarkerBackgroundColor: "#3B82F6",
    } as LineSeriesPartialOptions);
    macdRef.current = macdSeries;

    const signalSeries = chart.addLineSeries({
      color: "#F59E0B",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderColor: "#F59E0B",
      crosshairMarkerBackgroundColor: "#F59E0B",
    } as LineSeriesPartialOptions);
    signalRef.current = signalSeries;

    const histSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "histogram",
    } as HistogramSeriesPartialOptions);
    histRef.current = histSeries;

    chart.priceScale("histogram").applyOptions({
      scaleMargins: { top: 0.7, bottom: 0 },
    });

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
  }, [height]);

  useEffect(() => {
    if (!macdRef.current || !signalRef.current || !histRef.current || !macdData) return;

    macdRef.current.setData(macdData.macdLine);
    signalRef.current.setData(macdData.signalLine);

    const histBars = macdData.histogram.map((d) => ({
      time: d.time,
      value: Math.abs(d.value),
      color: d.value >= 0 ? "rgba(0, 200, 83, 0.6)" : "rgba(255, 82, 82, 0.6)",
    }));
    histRef.current.setData(histBars);
  }, [macdData]);

  if (loading) {
    return <div className="shimmer rounded-xl" style={{ height }} />;
  }

  if (!macdData || macdData.macdLine.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-xl border border-dashed border-neutral-700 bg-neutral-900/30" style={{ height }}>
        <p className="text-xs text-neutral-500">Insufficient data for MACD</p>
      </div>
    );
  }

  return (
    <div className="relative">
      <div ref={containerRef} />
      <img src="/logo.png" alt="" className="pointer-events-none ml-2 mb-1 size-8 opacity-60" />
      <div className="absolute top-2 left-3 flex items-center gap-3 pointer-events-none">
        <span className="flex items-center gap-1.5 text-[10px] text-neutral-500">
          <span className="size-2 rounded-full bg-accent-500" /> MACD
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-neutral-500">
          <span className="size-2 rounded-full bg-signal-neutral" /> Signal
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-neutral-500">
          <span className="size-2 rounded-full bg-signal-bullish" /> Bullish
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-neutral-500">
          <span className="size-2 rounded-full bg-signal-bearish" /> Bearish
        </span>
      </div>
    </div>
  );
}
