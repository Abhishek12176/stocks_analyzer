"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import { createChart, ColorType, type IChartApi, type ISeriesApi, type CandlestickSeriesPartialOptions, type HistogramSeriesPartialOptions, type LineSeriesPartialOptions } from "lightweight-charts";
import { cn } from "@/lib/cn";
import type { StockPrice } from "@/types/stock";

interface CandlestickChartProps {
  data: StockPrice[];
  volumeData?: { time: string; value: number }[];
  height?: number;
  loading?: boolean;
}

export function CandlestickChart({ data, volumeData, height = 400, loading }: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const sma20Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const sma50Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const toggleFullscreen = useCallback(() => {
    if (!wrapperRef.current) return;
    if (!document.fullscreenElement) {
      wrapperRef.current.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

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
        vertLines: { color: "#1F2937" },
        horzLines: { color: "#1F2937" },
      },
      crosshair: {
        mode: 0,
        vertLine: { color: "#374151", width: 1, style: 2, labelBackgroundColor: "#374151" },
        horzLine: { color: "#374151", width: 1, style: 2, labelBackgroundColor: "#374151" },
      },
      timeScale: {
        borderColor: "#1F2937",
        timeVisible: false,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: "#1F2937",
      },
      width: containerRef.current.clientWidth,
      height,
      handleScroll: true,
      handleScale: true,
    });

    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22C55E",
      downColor: "#EF4444",
      borderDownColor: "#EF4444",
      borderUpColor: "#22C55E",
      wickDownColor: "#EF4444",
      wickUpColor: "#22C55E",
    } as CandlestickSeriesPartialOptions);
    candlestickRef.current = candleSeries;

    const volSeries = chart.addHistogramSeries({
      color: "#3B82F6",
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    } as HistogramSeriesPartialOptions);
    volumeRef.current = volSeries;

    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    const line20 = chart.addLineSeries({
      color: "#F59E0B",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    } as LineSeriesPartialOptions);
    sma20Ref.current = line20;

    const line50 = chart.addLineSeries({
      color: "#60A5FA",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    } as LineSeriesPartialOptions);
    sma50Ref.current = line50;

    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chartRef.current = null;
      chart.remove();
    };
  }, [height]);

  useEffect(() => {
    if (!chartRef.current || !containerRef.current) return;
    const h = isFullscreen ? window.innerHeight - 160 : height;
    chartRef.current.applyOptions({ height: h, width: containerRef.current.clientWidth });
    chartRef.current.timeScale().fitContent();
  }, [isFullscreen, height]);

  useEffect(() => {
    if (!candlestickRef.current || !volumeRef.current) return;
    if (!data || data.length === 0) return;

    const candleData = data.map((d) => ({
      time: d.date.split("T")[0],
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));

    const formattedVolume = volumeData
      ? volumeData
      : data.map((d) => ({
          time: d.date.split("T")[0],
          value: d.volume,
        }));

    candlestickRef.current.setData(candleData);
    volumeRef.current.setData(formattedVolume);
  }, [data, volumeData]);

  useEffect(() => {
    if (!sma20Ref.current || !sma50Ref.current) return;
    if (!data || data.length === 0) return;

    const sma20Period = 20;
    const sma50Period = 50;

    if (data.length >= sma20Period) {
      const sma20Data = data.slice(sma20Period - 1).map((_, i) => {
        const slice = data.slice(i, i + sma20Period);
        const avg = slice.reduce((s, d) => s + d.close, 0) / sma20Period;
        return { time: data[i + sma20Period - 1].date.split("T")[0], value: avg };
      });
      sma20Ref.current.setData(sma20Data);
    }

    if (data.length >= sma50Period) {
      const sma50Data = data.slice(sma50Period - 1).map((_, i) => {
        const slice = data.slice(i, i + sma50Period);
        const avg = slice.reduce((s, d) => s + d.close, 0) / sma50Period;
        return { time: data[i + sma50Period - 1].date.split("T")[0], value: avg };
      });
      sma50Ref.current.setData(sma50Data);
    }
  }, [data]);

  return (
    <div
      ref={wrapperRef}
      className={cn("relative group", isFullscreen && "fixed inset-0 z-50 bg-[var(--color-bg)] p-6")}
    >
      {loading ? (
        <div className="shimmer rounded-lg" style={{ height }} />
      ) : data.length === 0 ? (
        <div
          className="flex items-center justify-center rounded-lg border border-dashed border-neutral-700 bg-neutral-900/50"
          style={{ height }}
        >
          <p className="text-sm text-neutral-500">No chart data available</p>
        </div>
      ) : null}
      <div
        ref={containerRef}
        className={cn(
          loading || data.length === 0 ? "hidden" : "",
          isFullscreen ? "h-full w-full" : ""
        )}
        style={!isFullscreen ? {} : { height: "calc(100vh - 160px)" }}
      />
      {!loading && data.length > 0 && (
        <button
          onClick={toggleFullscreen}
          className={cn(
            "absolute top-3 right-3 z-10 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all",
            "bg-neutral-800/80 text-neutral-400 hover:bg-neutral-700 hover:text-neutral-200",
            "opacity-0 group-hover:opacity-100",
            isFullscreen && "opacity-100"
          )}
        >
          <svg className="size-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            {isFullscreen ? (
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            )}
          </svg>
          {isFullscreen ? "Exit" : "Fullscreen"}
        </button>
      )}
    </div>
  );
}
