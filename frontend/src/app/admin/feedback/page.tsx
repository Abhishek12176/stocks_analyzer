"use client";

import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/lib/api";
import { motion } from "framer-motion";

interface FeedbackEntry {
  id: string;
  name: string;
  email: string;
  message: string;
  rating: number | null;
  created_at: string;
}

function StarRating({ rating }: { rating: number }) {
  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((star) => (
        <svg
          key={star}
          className={`size-3.5 ${star <= rating ? "text-accent-400" : "text-neutral-700"}`}
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
        </svg>
      ))}
    </div>
  );
}

export default function AdminFeedbackPage() {
  const { data, isLoading, error } = useQuery<FeedbackEntry[]>({
    queryKey: ["admin", "feedback"],
    queryFn: () => apiGet<FeedbackEntry[]>("/feedback"),
    refetchInterval: 60_000,
  });

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <div className="border-b border-neutral-800 px-8 py-6">
        <h1 className="text-2xl font-semibold text-neutral-100">Admin / Feedback</h1>
        <p className="mt-1 text-sm text-neutral-500">
          View user-submitted feedback
        </p>
      </div>

      <div className="p-8 max-w-4xl">
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-28 rounded-xl bg-neutral-900/60 animate-pulse" />
            ))}
          </div>
        ) : error ? (
          <div className="rounded-xl border border-signal-bearish/20 bg-signal-bearish/5 p-6 text-center">
            <p className="text-sm text-signal-bearish">Failed to load feedback</p>
          </div>
        ) : !data || data.length === 0 ? (
          <div className="rounded-xl border border-dashed border-neutral-800 p-12 text-center">
            <svg className="size-10 mx-auto text-neutral-600 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <h3 className="text-base font-semibold text-neutral-500 mb-1">No feedback yet</h3>
            <p className="text-sm text-neutral-600">Submissions will appear here once users send feedback.</p>
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-xs text-neutral-500 mb-4">{data.length} submission{data.length !== 1 ? "s" : ""}</p>
            {data.toReversed().map((entry, i) => (
              <motion.div
                key={entry.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.03, duration: 0.3 }}
                className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-5"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <p className="text-sm font-semibold text-neutral-200">
                      {entry.name || "Anonymous"}
                    </p>
                    <p className="text-xs text-neutral-500 mt-0.5">
                      {entry.email ? (
                        <a href={`mailto:${entry.email}`} className="text-accent-500 hover:underline">{entry.email}</a>
                      ) : "No email"}
                      {entry.rating && (
                        <> &middot; <StarRating rating={entry.rating} /></>
                      )}
                    </p>
                  </div>
                  <span className="text-[11px] text-neutral-600 whitespace-nowrap">
                    {new Date(entry.created_at).toLocaleDateString("en-IN", {
                      day: "numeric", month: "short", year: "numeric",
                      hour: "2-digit", minute: "2-digit",
                    })}
                  </span>
                </div>
                <p className="text-sm text-neutral-400 leading-relaxed whitespace-pre-wrap">
                  {entry.message}
                </p>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}
