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

export const screenerSchema = z.object({
  minROE: z.number().min(0).max(100).optional(),
  maxDE: z.number().min(0).max(10).optional(),
  minRevenueGrowth: z.number().min(-100).max(1000).optional(),
  minProfitGrowth: z.number().min(-100).max(1000).optional(),
  minOPM: z.number().min(0).max(100).optional(),
  maxPE: z.number().min(0).max(200).optional(),
  sector: z.string().optional(),
});

export type ScreenerFormData = z.infer<typeof screenerSchema>;
