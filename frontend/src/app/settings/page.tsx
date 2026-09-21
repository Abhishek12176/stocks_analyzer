"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { useUiStore } from "@/store/uiStore";
import { apiPost, apiGet } from "@/lib/api";

interface ApiKeyInfo {
  field: string;
  label: string;
  display: string;
  help: string;
  configured: boolean;
  masked: string | null;
}

export default function SettingsPage() {
  const { theme, setTheme } = useUiStore();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [rating, setRating] = useState(0);
  const [submitted, setSubmitted] = useState(false);
  const [sending, setSending] = useState(false);

  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[]>([]);
  const [keysLoading, setKeysLoading] = useState(true);
  const [keysError, setKeysError] = useState<string | null>(null);
  const [keyInputs, setKeyInputs] = useState<Record<string, string>>({});
  const [savingKeys, setSavingKeys] = useState(false);
  const [keysSaved, setKeysSaved] = useState(false);

  const loadApiKeys = async () => {
    setKeysLoading(true);
    setKeysError(null);
    try {
      const res = await apiGet<{ keys: ApiKeyInfo[] }>("/settings/api-keys");
      setApiKeys(res.keys);
    } catch {
      setKeysError("Could not load API key status.");
    } finally {
      setKeysLoading(false);
    }
  };

  useEffect(() => {
    loadApiKeys();
  }, []);

  const handleSaveKeys = async () => {
    const changed = Object.entries(keyInputs).filter(([, v]) => v.trim().length > 0);
    if (changed.length === 0) {
      setKeysError("Type a key value first, or clear the inputs and cancel.");
      return;
    }
    setSavingKeys(true);
    setKeysError(null);
    setKeysSaved(false);
    const payload: Record<string, string> = {};
    changed.forEach(([field, value]) => {
      payload[field] = value;
    });
    try {
      const res = await apiPost<{ keys: ApiKeyInfo[] }>("/settings/api-keys", payload);
      setApiKeys(res.keys);
      setKeyInputs({});
      setKeysSaved(true);
      setTimeout(() => setKeysSaved(false), 4000);
    } catch (e) {
      setKeysError(e instanceof Error ? e.message : "Failed to save API key.");
    } finally {
      setSavingKeys(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;
    setSending(true);
    try {
      await apiPost("/feedback", { name, email, message, rating: rating || null });
      setSubmitted(true);
    } catch {
      alert("Failed to send feedback. Try again.");
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="animate-fade-in">
      <div className="border-b border-neutral-800 px-8 py-6">
        <h1 className="text-2xl font-semibold text-neutral-100">Settings</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Configure your preferences
        </p>
      </div>

      <div className="p-8 max-w-2xl space-y-8">
        {/* Theme */}
        <section>
          <h2 className="text-base font-semibold text-neutral-200 mb-4">
            Appearance
          </h2>
          <div className="rounded-xl border border-neutral-800 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-neutral-300">Theme</p>
                <p className="text-xs text-neutral-500 mt-0.5">
                  Choose between dark and light mode
                </p>
              </div>
              <div className="flex gap-2">
                <Button
                  variant={theme === "dark" ? "primary" : "secondary"}
                  size="sm"
                  onClick={() => setTheme("dark")}
                >
                  Dark
                </Button>
                <Button
                  variant={theme === "light" ? "primary" : "secondary"}
                  size="sm"
                  onClick={() => setTheme("light")}
                >
                  Light
                </Button>
              </div>
            </div>
          </div>
        </section>

        {/* API Keys */}
        <section>
          <h2 className="text-base font-semibold text-neutral-200 mb-4">
            API Keys
          </h2>
          <div className="rounded-xl border border-neutral-800 p-5 space-y-4">
            {keysLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-14 w-full" />
                <Skeleton className="h-14 w-full" />
              </div>
            ) : keysError && apiKeys.length === 0 ? (
              <div className="flex flex-col items-start gap-3">
                <p className="text-sm text-signal-bearish">{keysError}</p>
                <Button variant="secondary" size="sm" onClick={loadApiKeys}>
                  Retry
                </Button>
              </div>
            ) : (
              <>
                <p className="text-xs text-neutral-500 leading-relaxed">
                  Keys are stored in <code className="text-neutral-400">backend/.env</code> and
                  are never returned to the browser. Configured keys take effect immediately
                  without restarting the server.
                </p>

                {apiKeys.map((key) => (
                  <div
                    key={key.field}
                    className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-4"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium text-neutral-200">{key.display}</p>
                        <p className="text-xs text-neutral-500 mt-0.5">{key.help}</p>
                      </div>
                      {key.configured ? (
                        <Badge variant="bullish">{key.masked}</Badge>
                      ) : (
                        <Badge variant="neutral">Not configured</Badge>
                      )}
                    </div>
                    <div className="mt-3 flex gap-2">
                      <input
                        type="password"
                        autoComplete="off"
                        value={keyInputs[key.field] ?? ""}
                        onChange={(e) =>
                          setKeyInputs((prev) => ({ ...prev, [key.field]: e.target.value }))
                        }
                        placeholder={
                          key.configured
                            ? "Paste a new key to replace the current one"
                            : "Paste your API key"
                        }
                        className="flex-1 rounded-lg border border-neutral-800 bg-neutral-900/60 px-3 py-2 text-sm text-neutral-50 placeholder-neutral-600 focus:outline-none focus:border-accent-500/40 transition-colors"
                      />
                    </div>
                  </div>
                ))}

                <div className="flex items-center gap-3 pt-1">
                  <Button
                    variant="primary"
                    size="sm"
                    loading={savingKeys}
                    onClick={handleSaveKeys}
                  >
                    Save Keys
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setKeyInputs({})}
                  >
                    Clear
                  </Button>
                  {keysSaved && (
                    <span className="text-xs text-signal-bullish">
                      Keys saved successfully
                    </span>
                  )}
                </div>

                {keysError && apiKeys.length > 0 && (
                  <p className="text-sm text-signal-bearish">{keysError}</p>
                )}
              </>
            )}
          </div>
        </section>

        {/* Feedback */}
        <section>
          <h2 className="text-base font-semibold text-neutral-200 mb-4">
            Feedback & Suggestions
          </h2>
          <div className="rounded-xl border border-neutral-800 p-5">
            {submitted ? (
              <div className="text-center py-6">
                <div className="size-12 mx-auto mb-3 rounded-full bg-signal-bullish/15 flex items-center justify-center">
                  <svg className="size-6 text-signal-bullish" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <p className="text-sm font-semibold text-neutral-200 mb-1">Thank you!</p>
                <p className="text-xs text-neutral-500">Your feedback helps make AVORA better.</p>
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-4"
                  onClick={() => { setSubmitted(false); setName(""); setEmail(""); setMessage(""); setRating(0); }}
                >
                  Send another
                </Button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-neutral-400 mb-1.5">Name</label>
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Your name"
                      className="w-full rounded-lg border border-neutral-800 bg-neutral-900/60 px-3 py-2 text-sm text-neutral-50 placeholder-neutral-600 focus:outline-none focus:border-accent-500/40 transition-colors"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-neutral-400 mb-1.5">Email</label>
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="you@example.com"
                      className="w-full rounded-lg border border-neutral-800 bg-neutral-900/60 px-3 py-2 text-sm text-neutral-50 placeholder-neutral-600 focus:outline-none focus:border-accent-500/40 transition-colors"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5">Message</label>
                  <textarea
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Tell us what you think — features, bugs, suggestions..."
                    rows={4}
                    className="w-full rounded-lg border border-neutral-800 bg-neutral-900/60 px-3 py-2 text-sm text-neutral-50 placeholder-neutral-600 focus:outline-none focus:border-accent-500/40 transition-colors resize-none"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5">Rating</label>
                  <div className="flex gap-1.5">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        type="button"
                        onClick={() => setRating(star)}
                        className={`size-8 rounded-lg flex items-center justify-center transition-all ${
                          star <= rating
                            ? "bg-accent-500/20 text-accent-400"
                            : "bg-neutral-800/50 text-neutral-600 hover:text-neutral-400"
                        }`}
                      >
                        <svg className="size-4" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                        </svg>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="pt-2">
                  <Button
                    type="submit"
                    variant="primary"
                    size="sm"
                    disabled={!message.trim() || sending}
                  >
                    {sending ? "Sending..." : "Send Feedback"}
                  </Button>
                </div>
              </form>
            )}
          </div>
        </section>

        {/* About */}
        <section>
          <h2 className="text-base font-semibold text-neutral-200 mb-4">
            About
          </h2>
          <div className="rounded-xl border border-neutral-800 p-5">
            <p className="text-sm text-neutral-500">
              AVORA v0.1.0 — Premium Indian Stock Analysis Platform
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
