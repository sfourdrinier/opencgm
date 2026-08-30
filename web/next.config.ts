// web/next.config.ts

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
    "/api/v1/**": [
      "./public/models/**",
      // onnxruntime_binding.node is traced because it is required; the shared library it
      // dlopens beside itself is not, because nothing imports it. Without this the function
      // deploys with the binding and no runtime behind it, and fails on the first call.
      "node_modules/onnxruntime-node/bin/napi-v*/linux/x64/libonnxruntime.so*",
    ],
  },

  // onnxruntime-node ships prebuilt binaries for every platform it supports -- 503 MB of
  // them -- and the tracer packs the lot, which puts the function over Vercel's 500 MB
  // limit on its own. A Linux x64 serverless function needs exactly one of these files.
  // The CUDA provider alone is 219 MB and there is no GPU to use it.
  outputFileTracingExcludes: {
    "/api/v1/**": [
      "node_modules/onnxruntime-node/bin/napi-v*/win32/**",
      "node_modules/onnxruntime-node/bin/napi-v*/darwin/**",
      "node_modules/onnxruntime-node/bin/napi-v*/linux/arm64/**",
      "node_modules/onnxruntime-node/bin/napi-v*/**/libonnxruntime_providers_cuda.so",
      "node_modules/onnxruntime-node/bin/napi-v*/**/libonnxruntime_providers_tensorrt.so",
      // Browser build. The demo loads it from the client bundle; a server function never
      // imports it, but it is 137 MB if it slips in.
      "node_modules/onnxruntime-web/**",
    ],
  },

  async headers() {
    const scriptSources = ["'self'", "'unsafe-inline'", "'wasm-unsafe-eval'"];
    if (process.env.NODE_ENV === "development") scriptSources.push("'unsafe-eval'");
    const contentSecurityPolicy = [
      "default-src 'self'",
      `script-src ${scriptSources.join(" ")}`,
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self'",
      "connect-src 'self'",
      "worker-src 'self' blob:",
      "object-src 'none'",
      "base-uri 'self'",
      "form-action 'self'",
      "frame-ancestors 'none'",
    ].join("; ");
    return [
      {
        source: "/:path*",
        headers: [{ key: "Content-Security-Policy", value: contentSecurityPolicy }],
      },
      {
        source: "/sensor/sensor-engine.abi1.wasm",
        headers: [{ key: "Cache-Control", value: "public, max-age=0, must-revalidate" }],
      },
    ];
  },
};

export default config;
