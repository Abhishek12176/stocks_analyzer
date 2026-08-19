"use client";

import { cn } from "@/lib/cn";

interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
  count?: number;
}

interface TabsProps {
  tabs: Tab[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}

export function Tabs({ tabs, active, onChange, className }: TabsProps) {
  return (
    <div className={cn("flex gap-1", className)}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={cn(
            "relative flex items-center gap-2 px-5 py-3 text-sm font-medium transition-all duration-200",
            active === tab.id
              ? "text-neutral-50"
              : "text-neutral-500 hover:text-neutral-300"
          )}
        >
          {tab.icon ? <span className="size-4">{tab.icon}</span> : null}
          {tab.label}
          {tab.count !== undefined ? (
            <span
              className={cn(
                "ml-1 rounded-md px-1.5 py-0.5 text-[11px] font-semibold",
                active === tab.id
                  ? "bg-accent-500/10 text-accent-400"
                  : "bg-neutral-800/50 text-neutral-500"
              )}
            >
              {tab.count}
            </span>
          ) : null}
          {active === tab.id ? (
            <div className="absolute bottom-0 left-3 right-3 h-[2px] bg-gradient-to-r from-accent-500 to-accent-400 rounded-full shadow-[0_0_8px_rgba(59,130,246,0.3)]" />
          ) : null}
        </button>
      ))}
    </div>
  );
}
