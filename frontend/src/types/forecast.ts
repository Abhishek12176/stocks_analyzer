export interface ForecastLatest {
  asOf: string;
  close?: number | null;
  rawProbability?: number | null;
  probability?: number | null;
  signal?: "BUY" | "HOLD" | "SELL" | null;
  direction: number;
  confidence?: number | null;
  reason?: string | null;
  thresholdBuy?: number | null;
  thresholdSell?: number | null;
}

export interface CalibrationInfo {
  method: string;
  n: number;
  nHoldout?: number | null;
  brier?: number | null;
  brierCalibrated?: number | null;
  ece?: number | null;
  eceCalibrated?: number | null;
  brierFit?: number | null;
  brierCalibratedFit?: number | null;
  eceFit?: number | null;
  eceCalibratedFit?: number | null;
  report?: Record<string, unknown>;
}

export interface EffectiveIndependentWindows {
  oof?: number | null;
  holdout?: number | null;
}

export interface DiscriminationInfo {
  metric?: string;
  metricDefinition?: string;
  fullOofAuc?: number | null;
  holdoutAuc?: number | null;
  fullOofAccuracyRaw?: number | null;
  holdoutAccuracyCalibrated?: number | null;
  holdoutAccuracyRaw?: number | null;
  holdoutUpRate?: number | null;
  holdoutBaselineAccuracy?: Record<string, number> | null;
  holdoutBestBaselineAccuracy?: number | null;
  holdoutEdgeVsBestBaselineAccuracy?: number | null;
  baselineRule?: string;
  effectiveIndependentWindows?: EffectiveIndependentWindows;
  universeNote?: string;
}

export interface ValidationInfo {
  targetFormula?: string;
  entryConvention?: string;
  exitConvention?: string;
  positionSizing?: string;
  maxConcurrentPositions?: string;
  oofRows?: number;
  calibrationFitRows?: number;
  holdoutRows?: number;
  oofRowsEffectiveIndependent?: number;
  holdoutRowsEffectiveIndependent?: number;
  effectiveSampleNote?: string;
  accuracyNote?: string;
  finalHoldoutNeverUsedForFitting?: boolean;
  finalHoldoutUsedFor?: string;
  isotonicCalibrationFitWindow?: string;
  auditReplayScope?: string;
}

export interface InputFingerprint {
  dataFingerprint?: string;
  algorithm?: string;
  scope?: string;
  asOf?: string;
  sources?: Record<string, unknown>;
  reconstruction?: string;
}

export interface ForecastExplanation {
  model: string;
  baselineProb: number;
  deltaScale: string;
  factors: { factor: string; impact: number; direction: "positive" | "negative" }[];
}

export interface ForecastBacktestRow {
  model: string;
  accuracy?: number | null;
  precision?: number | null;
  f1?: number | null;
  win_rate?: number | null;
  avg_return?: number | null;
  cum_return?: number | null;
  max_dd?: number | null;
  sharpe?: number | null;
  calmar?: number | null;
  profit_factor?: number | null;
  n_trades?: number | null;
}

export interface ForecastResponse {
  symbol: string;
  isAvailable: boolean;
  error?: string | null;
  horizon: number;
  generatedAt: string;
  versions?: Record<string, string>;
  latest?: ForecastLatest | null;
  calibration?: CalibrationInfo | null;
  discrimination?: DiscriminationInfo;
  validation?: ValidationInfo;
  dataFingerprint?: string | null;
  inputFingerprint?: InputFingerprint | null;
  backtest?: ForecastBacktestRow[];
  explanation?: ForecastExplanation | null;
  monitor?: {
    mode: string;
    freshness: string;
    max_feature_psi?: number | null;
    prediction_drift?: number | null;
    calibration_drift?: number | null;
    alarms?: string[];
  };
  snapshot?: Record<string, unknown>;
}

export interface MlSourceStatus {
  seriesId: string;
  name: string;
  isAvailable: boolean;
  rows: number;
  dataEnd?: string | null;
  ageDays?: number | null;
  status: string;
  error?: string | null;
}

export interface MlStatusResponse {
  status: string;
  mode: string;
  generatedAt: string;
  versions?: Record<string, string>;
  thresholds?: Record<string, number>;
  sources: MlSourceStatus[];
  lastForecast?: Record<string, unknown> | null;
  alarms: string[];
}