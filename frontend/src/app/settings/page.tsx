"use client";

import { Button } from "@/components/ui/Button";
import { useUiStore } from "@/store/uiStore";

export default function SettingsPage() {
  const { theme, setTheme } = useUiStore();

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

        {/* API Keys placeholder */}
        <section>
          <h2 className="text-base font-semibold text-neutral-200 mb-4">
            API Keys
          </h2>
          <div className="rounded-xl border border-neutral-800 p-5">
            <p className="text-sm text-neutral-500">
              API key configuration will be available here.
            </p>
          </div>
        </section>

        {/* About */}
        <section>
          <h2 className="text-base font-semibold text-neutral-200 mb-4">
            About
          </h2>
          <div className="rounded-xl border border-neutral-800 p-5">
            <p className="text-sm text-neutral-500">
              EquityLens v0.1.0 — Premium Indian Stock Analysis Platform
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
