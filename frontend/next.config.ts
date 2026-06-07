import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // API requests are proxied via src/app/api/[...path]/route.ts
  // which reads BACKEND_URL from environment variables.
  // This decouples the frontend build from the backend URL and
  // works identically in development and production.
};

export default nextConfig;
