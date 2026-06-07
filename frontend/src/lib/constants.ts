export const API_BASE = "/api/v1";

export const EXCHANGES = {
  NSE: ".NS",
  BSE: ".BO",
} as const;

export const DEFAULT_SYMBOL = "RELIANCE";

export const DATE_RANGES = [
  { label: "1M", days: 30 },
  { label: "3M", days: 90 },
  { label: "6M", days: 180 },
  { label: "1Y", days: 365 },
  { label: "MAX", days: 0 },
] as const;

export const SIGNAL_COLORS = {
  bullish: {
    text: "text-signal-bullish",
    bg: "bg-signal-bullish-bg",
    border: "border-signal-bullish/30",
  },
  bearish: {
    text: "text-signal-bearish",
    bg: "bg-signal-bearish-bg",
    border: "border-signal-bearish/30",
  },
  neutral: {
    text: "text-signal-neutral",
    bg: "bg-signal-neutral-bg",
    border: "border-signal-neutral/30",
  },
} as const;

export const INTRADAY_INTERVALS = [
  { label: "1m", value: "1m" },
  { label: "5m", value: "5m" },
  { label: "10m", value: "10m" },
  { label: "20m", value: "20m" },
  { label: "30m", value: "30m" },
  { label: "1h", value: "1h" },
] as const;

export const NAV_SECTIONS = [
  {
    label: "Overview",
    items: [
      { label: "Dashboard", href: "/", icon: "LayoutDashboard" },
    ],
  },
  {
    label: "Markets",
    items: [
      { label: "Stocks", href: "/markets/stocks", icon: "TrendingUp" },
    ],
  },
  {
    label: "Analysis",
    items: [
      { label: "Screener", href: "/screener", icon: "Filter" },
      { label: "Compare", href: "/compare", icon: "BarChart3" },
    ],
  },
  {
    label: "AI Tools",
    items: [
      { label: "Signals", href: "/signals", icon: "Zap" },
    ],
  },
  {
    label: "Portfolio",
    items: [
      { label: "Watchlist", href: "/watchlist", icon: "Bookmark" },
      { label: "Basket", href: "/basket", icon: "Basket" },
      { label: "History", href: "/history", icon: "Clock" },
    ],
  },
  {
    label: "Settings",
    items: [
      { label: "Settings", href: "/settings", icon: "Settings" },
    ],
  },
] as const;

type NavItem = { label: string; href: string; icon: string };
export const NAV_ITEMS: NavItem[] = ([] as NavItem[]).concat(...NAV_SECTIONS.map((s) => [...s.items] as NavItem[]));
