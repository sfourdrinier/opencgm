// Copy the onnxruntime-web WASM artefacts into public/ort/ so the browser can fetch them
// from our own origin. Runs as `postinstall`; referenced by lib/model/loadOnnx.ts, which
// sets `ort.env.wasm.wasmPaths = "/ort/"`.
//
// Without this the runtime falls back to a CDN, which (a) is a third-party request we did
// not ask the visitor to make, and (b) breaks entirely if the CDN version drifts from the
// installed package.
import { copyFileSync, mkdirSync, readdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const dest = join(root, "public", "ort");

const candidates = [
  join(root, "node_modules", "onnxruntime-web", "dist"),
];

const src = candidates.find((p) => existsSync(p));
if (!src) {
  console.warn("[copy-ort-assets] onnxruntime-web/dist not found; skipping");
  process.exit(0);
}

mkdirSync(dest, { recursive: true });

// Single-threaded, non-proxied builds only. The threaded (.jsep / -threaded) variants need
// SharedArrayBuffer, which needs COOP/COEP headers we deliberately do not set.
const wanted = readdirSync(src).filter(
  (f) => f.endsWith(".wasm") || (f.startsWith("ort-wasm") && f.endsWith(".mjs")),
);

let n = 0;
for (const file of wanted) {
  copyFileSync(join(src, file), join(dest, file));
  n += 1;
}
console.log(`[copy-ort-assets] copied ${n} file(s) to public/ort/`);
