function ema(data: number[], period: number): number[] {
  const result: number[] = [];
  const k = 2 / (period + 1);
  const sma = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
  result.push(sma);
  for (let i = period; i < data.length; i++) {
    const val = data[i] * k + result[result.length - 1] * (1 - k);
    result.push(val);
  }
  return result;
}

function sma(data: number[], period: number): number[] {
  const result: number[] = [];
  for (let i = period - 1; i < data.length; i++) {
    const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
    result.push(sum / period);
  }
  return result;
}

export function calculateRSI(closes: number[], period = 14): { value: number }[] {
  if (closes.length < period + 1) return [];
  const changes: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    changes.push(closes[i] - closes[i - 1]);
  }
  const gains: number[] = [];
  const losses: number[] = [];
  for (const c of changes) {
    gains.push(c > 0 ? c : 0);
    losses.push(c < 0 ? -c : 0);
  }
  const avgGain = sma(gains, period);
  const avgLoss = sma(losses, period);
  const rsiValues: number[] = [];
  for (let i = 0; i < avgGain.length; i++) {
    if (avgLoss[i] === 0) {
      rsiValues.push(100);
    } else {
      const rs = avgGain[i] / avgLoss[i];
      rsiValues.push(100 - 100 / (1 + rs));
    }
  }
  return rsiValues.map((v) => ({ value: v }));
}

export function calculateMACD(closes: number[], dates: string[]): {
  macdLine: { time: string; value: number }[];
  signalLine: { time: string; value: number }[];
  histogram: { time: string; value: number }[];
} {
  if (closes.length < 35) return { macdLine: [], signalLine: [], histogram: [] };
  const fastEMA = ema(closes, 12);
  const slowEMA = ema(closes, 26);
  const offset = 26 - 12;
  const macdLine: number[] = [];
  for (let i = 0; i < slowEMA.length; i++) {
    macdLine.push(fastEMA[i + offset] - slowEMA[i]);
  }
  const signalLine = ema(macdLine, 9);
  const histogram: number[] = [];
  const signalOffset = macdLine.length - signalLine.length;
  for (let i = 0; i < signalLine.length; i++) {
    histogram.push(macdLine[i + signalOffset] - signalLine[i]);
  }
  const macdStartIdx = 26 - 1;
  const sigStartIdx = macdStartIdx + 8;
  const actualDates = dates.slice(sigStartIdx);
  const histDates = dates.slice(sigStartIdx);

  return {
    macdLine: macdLine.map((v, i) => ({
      time: dates[macdStartIdx + i]?.split("T")[0] ?? "",
      value: Math.round(v * 100) / 100,
    })),
    signalLine: signalLine.map((v, i) => ({
      time: actualDates[i]?.split("T")[0] ?? "",
      value: Math.round(v * 100) / 100,
    })),
    histogram: histogram.map((v, i) => ({
      time: histDates[i]?.split("T")[0] ?? "",
      value: Math.round(v * 100) / 100,
    })),
  };
}
