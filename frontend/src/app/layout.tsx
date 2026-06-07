import type { Metadata } from "next";
import "@/styles/globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "AVORA — Stock Analysis Platform",
  description:
    "Premium Indian stock analysis platform combining technical, fundamental, ownership, and news data into actionable trade signals.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="antialiased">
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                var t = JSON.parse(localStorage.getItem("equitylens-ui"));
                if (t && t.state && t.state.theme === "light") {
                  document.documentElement.classList.remove("dark");
                  document.documentElement.classList.add("light");
                }
              } catch(e) {}
            `,
          }}
        />
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
