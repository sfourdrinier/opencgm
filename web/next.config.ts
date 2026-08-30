import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // `next dev -H 0.0.0.0` serves on every interface, but Next 16 refuses to hand its dev
  // chunks to a request whose Host is not localhost. The page then renders its server HTML
  // and never hydrates, which looks like "the interactive half of the site is missing".
  // These entries are development-only and have no effect on a production build.
  // Extra hosts come from the environment rather than the file: viewing the dev server from
  // a phone or another machine needs this box's LAN address, and that address has no business
  // in a public repository.
  //   NEXT_DEV_ORIGINS=192.168.1.20,mylaptop.local npm run dev
  allowedDevOrigins: [
    "127.0.0.1",
    "localhost",
    "*.local",
    ...(process.env.NEXT_DEV_ORIGINS?.split(",").map((s) => s.trim()).filter(Boolean) ?? []),
  ],
  // The /try demo runs ONNX Runtime Web in the browser. The /api/v1 routes additionally run
  // the same graph server-side through onnxruntime-node, for callers who want HTTP.
  // We deliberately do NOT enable `output: "standalone"` because the site
  // stays small and Vercel's default Next.js adapter handles it well.
  //
  // Multi-threaded WASM (SharedArrayBuffer) would need COOP/COEP headers. We do NOT
  // enable those — the single-thread WASM backend is fast enough for a 435,633-parameter
  // encoder and avoids the cross-origin-isolation dance entirely.
  // The API routes read encoder.onnx, encoder.meta.json and heads.json from public/ at
  // request time. A serverless bundler traces imports, not runtime readFileSync, so without
  // this the functions deploy without their model and fail with ENOENT on the first call --
  // while the static pages, which read the same files at build time, look perfectly fine.
  outputFileTracingIncludes: {
    "/api/v1/**": ["./public/models/**"],
  },

  async headers() {
    return [];
  },
};

export default config;
