"use client";

import { cn } from "@/lib/cn";

interface Column<T> {
  key: string;
  header: string;
  render: (item: T) => React.ReactNode;
  align?: "left" | "right";
  className?: string;
}

interface TableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor: (item: T) => string;
  className?: string;
  loading?: boolean;
  emptyMessage?: string;
}

export function Table<T>({
  columns,
  data,
  keyExtractor,
  className,
  loading = false,
  emptyMessage = "No data",
}: TableProps<T>) {
  if (loading) {
    return (
      <div className={cn("rounded-xl border border-neutral-800", className)}>
        <div className="border-b border-neutral-800 px-4 py-3 flex gap-6">
          {columns.map((col) => (
            <div key={col.key} className="shimmer h-4 w-24 rounded" />
          ))}
        </div>
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="border-b border-neutral-800/50 px-4 py-3 flex gap-6">
            {columns.map((col) => (
              <div key={col.key} className="shimmer h-4 w-20 rounded" />
            ))}
          </div>
        ))}
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div
        className={cn(
          "rounded-xl border border-neutral-800 py-12 text-center text-sm text-neutral-500",
          className
        )}
      >
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className={cn("rounded-xl border border-neutral-800 overflow-hidden", className)}>
      <table className="w-full">
        <thead>
          <tr className="border-b border-neutral-800">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  "px-4 py-3 text-xs font-medium text-neutral-500 uppercase tracking-wider",
                  col.align === "right" && "text-right",
                  col.className
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((item) => (
            <tr
              key={keyExtractor(item)}
              className="border-b border-neutral-800/50 transition-colors hover:bg-neutral-800/30"
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={cn(
                    "px-4 py-3 text-sm text-neutral-300",
                    col.align === "right" && "text-right font-mono",
                    col.className
                  )}
                >
                  {col.render(item)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
