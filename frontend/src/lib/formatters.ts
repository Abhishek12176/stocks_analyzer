export function formatIndianNumber(value: number | null | undefined): string {
  if (value == null) return "N/A";
  if (value >= 1e7) return `₹${(value / 1e7).toFixed(2)} Cr`;
  if (value >= 1e5) return `₹${(value / 1e5).toFixed(2)} L`;
  return `₹${value.toLocaleString("en-IN")}`;
}

export function formatCrore(value: number | null | undefined): string {
  if (value == null) return "N/A";
  return `₹${(value / 1e7).toFixed(2)} Cr`;
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null) return "N/A";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

export function formatPrice(value: number | null | undefined): string {
  if (value == null) return "N/A";
  return `₹${value.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function formatRatio(value: number | null | undefined): string {
  if (value == null) return "N/A";
  return value.toFixed(2);
}

export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "Latest";
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-IN", {
      day: "numeric",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "Latest";
  }
}

export function formatCompactNumber(value: number | null | undefined): string {
  if (value == null) return "N/A";
  if (value >= 1e7) return `${(value / 1e7).toFixed(1)}Cr`;
  if (value >= 1e5) return `${(value / 1e5).toFixed(1)}L`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
  return value.toLocaleString("en-IN");
}
