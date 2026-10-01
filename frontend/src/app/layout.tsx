import type { Metadata } from "next";
import "@/styles/globals.css";
import { Providers } from "./providers";


const getBaseUrl = (): string => {
  if (process.env.NEXT_PUBLIC_SITE_URL) {
    return process.env.NEXT_PUBLIC_SITE_URL.replace(/\/$/, "");
  }
  if (process.env.VERCEL_PROJECT_PRODUCTION_URL) {
    return `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`;
  }
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`;
  }
  return "https://equitylens.vercel.app";
};

export const metadata: Metadata = {
  metadataBase: new URL(getBaseUrl()),
  title: "AVORA — Stock Analysis Platform",
  description:
    "Premium Indian stock analysis platform combining technical, fundamental, ownership, and news data into actionable trade signals.",
  icons: {
    icon: "/logo.png",
    apple: "/logo.png",
  },
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
