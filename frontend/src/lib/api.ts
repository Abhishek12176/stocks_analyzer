import { API_BASE } from "./constants";
import type { ApiError } from "@/types/api";

export class ApiClientError extends Error {
  constructor(
    public code: string,
    message: string,
    public details?: string
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const apiError = body as ApiError;
    throw new ApiClientError(
      apiError.code || "UNKNOWN",
      apiError.error || `HTTP ${response.status}`,
      apiError.details
    );
  }
  return response.json();
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, string>
): Promise<T> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value);
    });
  }
  const response = await fetch(url.toString());
  return handleResponse<T>(response);
}

export async function apiPost<T>(
  path: string,
  body: unknown
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleResponse<T>(response);
}

export async function apiDelete(path: string): Promise<void> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, { method: "DELETE" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const apiError = body as ApiError;
    throw new ApiClientError(
      apiError.code || "UNKNOWN",
      apiError.error || `HTTP ${response.status}`,
      apiError.details
    );
  }
}
