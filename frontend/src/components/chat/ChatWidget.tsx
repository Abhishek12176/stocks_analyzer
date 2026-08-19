"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { sendChatMessage } from "@/lib/chat-api";
import { cn } from "@/lib/cn";
import { StockResultCard } from "./StockResultCard";
import type { ChatStock, ChatTurn } from "@/types/chat";

interface WidgetMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  stocks?: ChatStock[];
}

const QUICK_PROMPTS = [
  "Stocks under ₹500",
  "mujhe 500 se kam rate wale stock predictions do",
  "Top buy signals",
  "Sell signals abhi",
];

const WELCOME_TEXT =
  "Namaste! 👋 Main AVORA ka AI assistant hoon. Main aapko NSE stocks ke **20-din ke predictions** bata sakta hoon.\n\nTry karein: **500 se kam rate wale stocks** ya **top buy signals**.";

let messageCounter = 0;
const nextId = () => `msg-${Date.now()}-${messageCounter++}`;

function renderInline(text: string, symbols: string[]) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      const inner = part.slice(2, -2).trim();
      const symbol = symbols.find((s) => inner.toUpperCase() === s.toUpperCase());
      if (symbol) {
        return (
          <Link
            key={i}
            href={`/stock/${symbol}?exchange=NSE`}
            className="font-bold text-accent-400 underline decoration-accent-500/40 underline-offset-2 hover:text-accent-300 transition-colors"
          >
            {inner}
          </Link>
        );
      }
      return (
        <strong key={i} className="font-bold text-neutral-100">
          {inner}
        </strong>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

function AssistantMessage({ msg }: { msg: WidgetMessage }) {
  const symbols = (msg.stocks ?? []).map((s) => s.symbol);
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-tl-sm border border-neutral-800 bg-neutral-900/80 px-3.5 py-2.5">
        <div className="whitespace-pre-line text-[13px] leading-relaxed text-neutral-300">
          {renderInline(msg.content, symbols)}
        </div>
        {msg.stocks && msg.stocks.length > 0 && (
          <div className="mt-2.5 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {msg.stocks.map((stock, i) => (
              <StockResultCard key={stock.symbol} stock={stock} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function UserMessage({ msg }: { msg: WidgetMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-accent-500/15 border border-accent-500/20 px-3.5 py-2.5">
        <p className="text-[13px] leading-relaxed text-neutral-100 whitespace-pre-line">
          {msg.content}
        </p>
      </div>
    </div>
  );
}

export function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<WidgetMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setTimeout(() => inputRef.current?.focus(), 80);
  }, [open]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading, open]);

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    setError(null);
    setInput("");

    const history: ChatTurn[] = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    const userMsg: WidgetMessage = { id: nextId(), role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await sendChatMessage(trimmed, history);
      const assistantMsg: WidgetMessage = {
        id: nextId(),
        role: "assistant",
        content: res.reply,
        stocks: res.stocks,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(
        err instanceof Error && err.message
          ? err.message
          : "Kuch gadbad ho gayi. Dobara try karein."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating launcher button */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="fixed bottom-6 right-6 z-50 flex size-14 items-center justify-center rounded-full bg-accent-500 text-white shadow-[0_8px_30px_rgba(59,130,246,0.35)] transition-all duration-200 hover:scale-105 hover:shadow-[0_8px_40px_rgba(59,130,246,0.5)]"
        aria-label={open ? "Close AI chat" : "Open AI chat"}
      >
        <AnimatePresence mode="wait" initial={false}>
          {open ? (
            <motion.svg
              key="close"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="size-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </motion.svg>
          ) : (
            <motion.svg
              key="chat"
              initial={{ scale: 0.6, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.6, opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="size-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
              />
            </motion.svg>
          )}
        </AnimatePresence>
      </button>

      {/* Chat panel */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.97 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="fixed bottom-24 right-6 z-50 flex h-[560px] w-[calc(100vw-3rem)] max-w-[380px] flex-col overflow-hidden rounded-2xl border border-neutral-800 bg-neutral-950/95 shadow-2xl backdrop-blur-xl"
          >
            {/* Header */}
            <div className="flex items-center gap-3 border-b border-neutral-800 px-4 py-3 bg-neutral-900/80">
              <div className="relative flex size-9 items-center justify-center rounded-xl bg-accent-500/15 border border-accent-500/25">
                <span className="font-mono text-sm font-bold text-accent-400">AI</span>
                <span className="absolute -right-0.5 -top-0.5 size-2 rounded-full bg-signal-bullish" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-neutral-50">AVORA AI Assistant</p>
                <p className="text-[11px] text-neutral-500">NSE stock predictions • 20-day outlook</p>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="flex size-7 items-center justify-center rounded-lg text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800 transition-colors"
                aria-label="Minimize chat"
              >
                <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14" />
                </svg>
              </button>
            </div>

            {/* Messages */}
            <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-3.5 py-4">
              {messages.length === 0 && (
                <div className="space-y-3">
                  <div className="flex justify-start">
                    <div className="max-w-[85%] rounded-2xl rounded-tl-sm border border-neutral-800 bg-neutral-900/80 px-3.5 py-2.5">
                      <div className="whitespace-pre-line text-[13px] leading-relaxed text-neutral-300">
                        {renderInline(WELCOME_TEXT, [])}
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {QUICK_PROMPTS.map((prompt) => (
                      <button
                        key={prompt}
                        onClick={() => send(prompt)}
                        className="rounded-full border border-neutral-800 bg-neutral-900/70 px-2.5 py-1 text-[11px] text-neutral-400 transition-colors hover:border-accent-500/40 hover:text-accent-400"
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg) =>
                msg.role === "user" ? (
                  <UserMessage key={msg.id} msg={msg} />
                ) : (
                  <AssistantMessage key={msg.id} msg={msg} />
                )
              )}

              {loading && (
                <div className="flex justify-start">
                  <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm border border-neutral-800 bg-neutral-900/80 px-3.5 py-3">
                    <span className="size-1.5 rounded-full bg-neutral-500 animate-bounce" />
                    <span className="size-1.5 rounded-full bg-neutral-500 animate-bounce [animation-delay:120ms]" />
                    <span className="size-1.5 rounded-full bg-neutral-500 animate-bounce [animation-delay:240ms]" />
                  </div>
                </div>
              )}

              {error && (
                <div className="rounded-xl border border-signal-bearish/30 bg-signal-bearish/10 px-3 py-2.5 text-[12px] text-signal-bearish">
                  {error}
                </div>
              )}
            </div>

            {/* Input */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                send(input);
              }}
              className="flex items-center gap-2 border-t border-neutral-800 bg-neutral-900/80 px-3 py-2.5"
            >
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about stock predictions..."
                disabled={loading}
                className={cn(
                  "flex-1 rounded-xl border border-neutral-800 bg-neutral-950/70 px-3 py-2 text-[13px] text-neutral-100 placeholder-neutral-500 outline-none transition-colors",
                  "focus:border-accent-500/40 focus:ring-2 focus:ring-accent-500/10"
                )}
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent-500 text-white transition-all hover:bg-accent-600 disabled:opacity-40 disabled:hover:bg-accent-500"
                aria-label="Send message"
              >
                <svg className="size-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
