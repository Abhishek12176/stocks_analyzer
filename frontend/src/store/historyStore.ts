"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface HistoryItem {
  symbol: string;
  name: string;
  price: number | null;
  signal: string | null;
  score: number | null;
  visitedAt: string;
}

interface HistoryState {
  items: HistoryItem[];
  add: (item: Omit<HistoryItem, "visitedAt">) => void;
  clear: () => void;
}

const MAX_HISTORY = 50;

export const useHistoryStore = create<HistoryState>()(
  persist(
    (set) => ({
      items: [],
      add: (item) =>
        set((state) => {
          const filtered = state.items.filter(
            (i) => i.symbol !== item.symbol
          );
          return {
            items: [
              { ...item, visitedAt: new Date().toISOString() },
              ...filtered,
            ].slice(0, MAX_HISTORY),
          };
        }),
      clear: () => set({ items: [] }),
    }),
    { name: "equitylens-history" }
  )
);
