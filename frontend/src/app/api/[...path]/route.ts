import { NextRequest, NextResponse } from "next/server";

// In production, set BACKEND_URL on Vercel to https://stocks-analyzer-9fg7.onrender.com
// For local dev, create a .env.local file with BACKEND_URL=http://localhost:8000
const BACKEND_URL = process.env.BACKEND_URL || "https://stocks-analyzer-9fg7.onrender.com";

const FETCH_TIMEOUT = 30_000;

async function fetchWithTimeout(url: string, options: RequestInit = {}): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(timeout);
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const pathname = path.join("/");
  const searchParams = request.nextUrl.searchParams.toString();
  const url = `${BACKEND_URL}/api/${pathname}${searchParams ? `?${searchParams}` : ""}`;

  try {
    const response = await fetchWithTimeout(url, {
      headers: {
        "Content-Type": "application/json",
        ...(process.env.API_KEY
          ? { "X-API-Key": process.env.API_KEY }
          : {}),
      },
    });

    const data = await response.json();

    return NextResponse.json(data, {
      status: response.status,
      headers: {
        "Cache-Control": "public, s-maxage=60, stale-while-revalidate=30",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Backend service unavailable",
        code: "SERVICE_UNAVAILABLE",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 503 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const pathname = path.join("/");
  const url = `${BACKEND_URL}/api/${pathname}`;

  try {
    const body = await request.json();
    const response = await fetchWithTimeout(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.API_KEY
          ? { "X-API-Key": process.env.API_KEY }
          : {}),
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Backend service unavailable",
        code: "SERVICE_UNAVAILABLE",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 503 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const pathname = path.join("/");
  const url = `${BACKEND_URL}/api/${pathname}`;

  try {
    const response = await fetchWithTimeout(url, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.API_KEY
          ? { "X-API-Key": process.env.API_KEY }
          : {}),
      },
    });

    if (response.ok) {
      return new NextResponse(null, { status: 204 });
    }

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        error: "Backend service unavailable",
        code: "SERVICE_UNAVAILABLE",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 503 }
    );
  }
}
