"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import { createChart, ColorType, type IChartApi, type ISeriesApi, type CandlestickSeriesPartialOptions, type HistogramSeriesPartialOptions, type LineSeriesPartialOptions } from "lightweight-charts";
import { cn } from "@/lib/cn";
import { INTRADAY_INTERVALS } from "@/lib/constants";
import { formatPrice, formatPercent } from "@/lib/formatters";
import type { IntradayPrice } from "@/types/stock";

interface IntradayChartProps {
  symbol?: string;
  companyName?: string;
  currentPrice?: number;
  change?: number;
  changePercent?: number;
  lastUpdated?: string;
  data: IntradayPrice[];
  interval: string;
  onIntervalChange: (interval: string) => void;
  loading?: boolean;
  height?: number;
}

function generateDepthData(currentPrice: number) {
  const bids: { price: number; size: number }[] = [];
  const asks: { price: number; size: number }[] = [];
  for (let i = 1; i <= 8; i++) {
    const bidPrice = currentPrice - i * 0.15 - Math.random() * 0.1;
    const askPrice = currentPrice + i * 0.15 + Math.random() * 0.1;
    bids.push({
      price: parseFloat(bidPrice.toFixed(2)),
      size: Math.round(500 + Math.random() * 3000 - i * 100),
    });
    asks.push({
      price: parseFloat(askPrice.toFixed(2)),
      size: Math.round(500 + Math.random() * 3000 - i * 100),
    });
  }
  return { bids, asks };
}

function generateSignals(data: IntradayPrice[]): { time: string; type: "buy" | "sell" }[] {
  if (data.length < 10) return [];
  const signals: { time: string; type: "buy" | "sell" }[] = [];
  let consecutiveUp = 0;
  let consecutiveDown = 0;
  for (let i = 3; i < data.length; i++) {
    if (data[i].close > data[i].open) {
      consecutiveUp++;
      consecutiveDown = 0;
    } else {
      consecutiveDown++;
      consecutiveUp = 0;
    }
    if (consecutiveUp >= 3 && data[i].close > data[i - 1].close * 1.005) {
      signals.push({ time: data[i].time, type: "buy" });
      consecutiveUp = 0;
    }
    if (consecutiveDown >= 3 && data[i].close < data[i - 1].close * 0.995) {
      signals.push({ time: data[i].time, type: "sell" });
      consecutiveDown = 0;
    }
  }
  return signals;
}

export function IntradayChart({
  symbol,
  companyName,
  currentPrice,
  change,
  changePercent,
  lastUpdated,
  data,
  interval,
  onIntervalChange,
  loading,
  height = 480,
}: IntradayChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const vwapRef = useRef<ISeriesApi<"Line"> | null>(null);

  const [chartReady, setChartReady] = useState(false);

  const price = currentPrice ?? (data.length > 0 ? data[data.length - 1].close : 0);
  const priceChange = change;
  const priceChangePercent = changePercent;

  const isPositive = priceChange != null ? priceChange >= 0 : true;

  const depthData = useMemo(() => generateDepthData(price), [price]);
  const signals = useMemo(() => generateSignals(data), [data]);

  const totalBidSize = useMemo(() => depthData.bids.reduce((s, d) => s + d.size, 0), [depthData]);
  const totalAskSize = useMemo(() => depthData.asks.reduce((s, d) => s + d.size, 0), [depthData]);

  const maxDepthSize = useMemo(() => {
    const all = [...depthData.bids, ...depthData.asks];
    return Math.max(...all.map((d) => d.size), 1);
  }, [depthData]);

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
        vertLine: {
          color: "#374151",
          width: 1,
          style: 2,
          labelBackgroundColor: "#374151",
        },
        horzLine: {
          color: "#374151",
          width: 1,
          style: 2,
          labelBackgroundColor: "#374151",
        },
      },
      timeScale: {
        borderColor: "#1F2937",
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: number) => {
          const date = new Date(time * 1000);
          const hours = date.getHours().toString().padStart(2, "0");
          const mins = date.getMinutes().toString().padStart(2, "0");
          return `${hours}:${mins}`;
        },
      },
      rightPriceScale: {
        borderColor: "#1F2937",
        scaleMargins: { top: 0.05, bottom: 0.25 },
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

    const vwapLine = chart.addLineSeries({
      color: "#8B5CF6",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    } as LineSeriesPartialOptions);
    vwapRef.current = vwapLine;

    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    setChartReady(true);

    return () => {
      window.removeEventListener("resize", handleResize);
      chartRef.current = null;
      setChartReady(false);
      chart.remove();
    };
  }, [height]);

  useEffect(() => {
    if (!candlestickRef.current || !volumeRef.current || !vwapRef.current || !chartReady) return;
    if (!data || data.length === 0) return;

    const candleData = data.map((d) => ({
      time: Math.floor(new Date(d.time).getTime() / 1000) as any,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));

    const volData = data.map((d) => ({
      time: Math.floor(new Date(d.time).getTime() / 1000) as any,
      value: d.volume,
      color: d.close >= d.open ? "rgba(34, 197, 94, 0.4)" : "rgba(239, 68, 68, 0.4)",
    }));

    const vwapValues = data.map((d) => {
      const typical = (d.high + d.low + d.close) / 3;
      return typical;
    });

    let cumulativePV = 0;
    let cumulativeV = 0;
    const vwapData = data.map((d, i) => {
      const typical = (d.high + d.low + d.close) / 3;
      cumulativePV += typical * d.volume;
      cumulativeV += d.volume;
      const vwap = cumulativeV > 0 ? cumulativePV / cumulativeV : typical;
      return {
        time: Math.floor(new Date(d.time).getTime() / 1000) as any,
        value: parseFloat(vwap.toFixed(2)),
      };
    });

    candlestickRef.current.setData(candleData);
    volumeRef.current.setData(volData);
    vwapRef.current.setData(vwapData);

    if (signals.length > 0) {
      const markers = signals.map((s) => ({
        time: Math.floor(new Date(s.time).getTime() / 1000) as any,
        position: s.type === "buy" ? "belowBar" as const : "aboveBar" as const,
        color: s.type === "buy" ? "#22C55E" : "#EF4444",
        shape: s.type === "buy" ? "arrowUp" as const : "arrowDown" as const,
        text: s.type === "buy" ? "BUY" : "SELL",
      }));
      candlestickRef.current.setMarkers(markers);
    }

    chartRef.current?.timeScale().fitContent();
  }, [data, chartReady, signals]);

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900/50 overflow-hidden">
      <div className="px-5 py-4 border-b border-neutral-800">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-4">
            {symbol && (
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-lg font-semibold text-neutral-100">{symbol}</h3>
                  {companyName && (
                    <span className="text-xs text-neutral-500 hidden sm:inline">{companyName}</span>
                  )}
                </div>
                <div className="flex items-center gap-3 mt-0.5">
                  <span className="text-2xl font-mono font-bold text-neutral-100">
                    {formatPrice(price)}
                  </span>
                  {priceChange != null && (
                    <span className={cn(
                      "flex items-center gap-1 text-sm font-mono font-medium",
                      isPositive ? "text-signal-bullish" : "text-signal-bearish"
                    )}>
                      {isPositive ? "▲" : "▼"}
                      {formatPrice(Math.abs(priceChange))}
                      <span className="text-xs">
                        ({formatPercent(priceChangePercent ?? 0)})
                      </span>
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
          {lastUpdated && (
            <span className="text-xs text-neutral-500">
              Updated: {new Date(lastUpdated).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between px-5 py-2.5 border-b border-neutral-800 bg-neutral-900/80">
        <div className="flex items-center gap-1">
          {INTRADAY_INTERVALS.map((intv) => (
            <button
              key={intv.value}
              onClick={() => onIntervalChange(intv.value)}
              className={cn(
                "px-3 py-1.5 text-xs font-medium rounded-lg transition-colors",
                interval === intv.value
                  ? "bg-accent-500/10 text-accent-400 border border-accent-500/20"
                  : "text-neutral-500 hover:text-neutral-300 border border-transparent"
              )}
            >
              {intv.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 text-xs text-neutral-500">
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-signal-bullish" />
            Buy
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-full bg-signal-bearish" />
            Sell
          </span>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row">
        <div className="flex-1 min-w-0 relative">
          {loading ? (
            <div className="shimmer rounded-lg" style={{ height }} />
          ) : data.length === 0 ? (
            <div className="flex items-center justify-center border border-dashed border-neutral-700 bg-neutral-900/30 rounded-lg m-2" style={{ height }}>
              <p className="text-sm text-neutral-500">No intraday data available</p>
            </div>
          ) : null}
          <div
            ref={containerRef}
            className={cn(
              loading || data.length === 0 ? "hidden" : "",
              "w-full"
            )}
          />
        </div>

        <div className="w-full lg:w-56 xl:w-64 border-t lg:border-t-0 lg:border-l border-neutral-800 p-4 space-y-4">
          <div>
            <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-2">
              Market Depth
            </p>
            <div className="space-y-0.5">
              {depthData.asks.slice().reverse().map((ask) => (
                <div key={`ask-${ask.price}`} className="flex items-center justify-between text-xs relative h-5">
                  <div
                    className="absolute right-0 top-0 bottom-0 bg-signal-bearish/10 rounded"
                    style={{ width: `${(ask.size / maxDepthSize) * 100}%` }}
                  />
                  <span className="font-mono text-signal-bearish relative z-10">{ask.price.toFixed(2)}</span>
                  <span className="font-mono text-neutral-500 relative z-10">{ask.size}</span>
                </div>
              ))}
              <div className="flex items-center justify-between text-xs font-semibold py-1 border-y border-neutral-800 my-1">
                <span className="font-mono text-neutral-100">{price.toFixed(2)}</span>
                <span className="text-neutral-500 text-[10px]">
                  Spread: {(depthData.asks[0]?.price - depthData.bids[0]?.price).toFixed(2)}
                </span>
              </div>
              {depthData.bids.map((bid) => (
                <div key={`bid-${bid.price}`} className="flex items-center justify-between text-xs relative h-5">
                  <div
                    className="absolute left-0 top-0 bottom-0 bg-signal-bullish/10 rounded"
                    style={{ width: `${(bid.size / maxDepthSize) * 100}%` }}
                  />
                  <span className="font-mono text-signal-bullish relative z-10">{bid.price.toFixed(2)}</span>
                  <span className="font-mono text-neutral-500 relative z-10">{bid.size}</span>
                </div>
              ))}
            </div>
            <div className="flex items-center justify-between text-[10px] text-neutral-600 mt-1.5">
              <span>Bid: {totalBidSize.toLocaleString()}</span>
              <span>Ask: {totalAskSize.toLocaleString()}</span>
            </div>
          </div>

          <div>
            <p className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-2">
              Holdings
            </p>
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-neutral-500">Quantity</span>
                <span className="font-mono text-neutral-200">50</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-neutral-500">Entry</span>
                <span className="font-mono text-neutral-200">{formatPrice(price * 0.985)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-neutral-500">P&amp;L</span>
                <span className={cn("font-mono font-medium", isPositive ? "text-signal-bullish" : "text-signal-bearish")}>
                  {isPositive ? "+" : ""}{formatPrice(Math.abs(price * 0.015 * 50))}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-neutral-500">Day P&amp;L</span>
                {priceChange != null && (
                  <span className={cn("font-mono font-medium", isPositive ? "text-signal-bullish" : "text-signal-bearish")}>
                    {isPositive ? "+" : ""}{formatPrice(Math.abs(priceChange * 50))}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 px-5 py-3 border-t border-neutral-800 bg-neutral-900/80">
        <button className="flex-1 px-4 py-2 rounded-lg bg-signal-bullish/10 text-signal-bullish text-sm font-semibold hover:bg-signal-bullish/20 transition-colors">
          Buy
        </button>
        <button className="flex-1 px-4 py-2 rounded-lg bg-signal-bearish/10 text-signal-bearish text-sm font-semibold hover:bg-signal-bearish/20 transition-colors">
          Sell
        </button>
      </div>
    </div>
  );
}
