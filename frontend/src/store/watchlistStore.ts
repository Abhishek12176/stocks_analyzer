"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface WatchlistItem {
  symbol: string;
  name: string;
  addedAt: string;
}

interface WatchlistState {
  items: WatchlistItem[];
  add: (symbol: string, name: string) => void;
  remove: (symbol: string) => void;
  has: (symbol: string) => boolean;
  clear: () => void;
}

export const useWatchlistStore = create<WatchlistState>()(
  persist(
    (set, get) => ({
      items: [],
      add: (symbol, name) =>
        set((state) => {
          if (state.items.some((i) => i.symbol === symbol)) return state;
          return {
            items: [
              ...state.items,
              { symbol, name, addedAt: new Date().toISOString() },
            ],
          };
        }),
      remove: (symbol) =>
        set((state) => ({
          items: state.items.filter((i) => i.symbol !== symbol),
        })),
      has: (symbol) => get().items.some((i) => i.symbol === symbol),
      clear: () => set({ items: [] }),
    }),
    { name: "equitylens-watchlist" }
  )
);
