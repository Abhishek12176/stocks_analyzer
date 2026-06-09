import { z } from "zod";

export const symbolSchema = z
  .string()
  .min(1, "Symbol is required")
  .max(20, "Symbol is too long")
  .transform((s) => s.toUpperCase().replace(/\.NS|\.BO/g, "").trim())
  .refine((s) => /^[A-Z0-9]{1,10}$/.test(s), "Invalid stock symbol");

export const searchQuerySchema = z.string().min(1).max(100);

export const watchlistItemSchema = z.object({
  symbol: symbolSchema,
  addedAt: z.string().datetime().optional(),
});

export const compareSchema = z.object({
  symbols: z
    .array(symbolSchema)
    .min(2, "Select at least 2 stocks")
    .max(5, "Maximum 5 stocks"),
});


