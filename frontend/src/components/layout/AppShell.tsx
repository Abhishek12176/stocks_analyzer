"use client";

import { Sidebar } from "./Sidebar";
import { Header } from "./Header";
import { MobileNav } from "./MobileNav";
import { CommandPalette } from "./CommandPalette";
import { ChatWidget } from "@/components/chat/ChatWidget";
import { ToastContainer } from "@/components/ui/Toast";
import { cn } from "@/lib/cn";
import { useUiStore } from "@/store/uiStore";

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const collapsed = useUiStore((s) => s.sidebarCollapsed);

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div
        className={cn(
          "flex flex-1 flex-col transition-all duration-200",
          collapsed ? "ml-16" : "ml-60"
        )}
      >
        <Header />
        <main className="flex-1">{children}</main>
      </div>
      <MobileNav />
      <CommandPalette />
      <ChatWidget />
      <ToastContainer />
    </div>
  );
}
