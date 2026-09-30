import type { NextConfig } from "next";

// The browser talks to /api/axiom/*, which Next proxies to `axiom serve`, so the
// FastAPI service needs no CORS and stays bound to localhost.
const AXIOM_API_URL = process.env.AXIOM_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/axiom/:path*", destination: `${AXIOM_API_URL}/v1/:path*` }];
  },
};

export default nextConfig;
