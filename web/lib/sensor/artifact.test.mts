// web/lib/sensor/artifact.test.mts

import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import assert from "node:assert/strict";
import { verifySensorArtifact } from "../../scripts/verify-sensor-artifact";

const publicRoot = resolve(import.meta.dirname, "../../public/sensor");
const artifactPath = resolve(publicRoot, "sensor-engine.abi1.wasm");
const manifestPath = resolve(publicRoot, "sensor-engine.manifest.json");

type ArtifactManifest = {
  readonly abiVersion: number;
  readonly filename: string;
  readonly byteLength: number;
  readonly sha256: string;
  readonly exports: readonly string[];
  readonly maximumInputBytes: number;
  readonly maximumOutputBytes: number;
  readonly toolchain: Record<string, string>;
  readonly licenses: readonly string[];
};

test("ships exactly one audited ABI 1 sensor artifact and matching manifest", async () => {
  const artifact = await readFile(artifactPath);
  const manifestText = await readFile(manifestPath, "utf8");
  const entries = await readdir(publicRoot);
  const parsed: unknown = JSON.parse(manifestText);
  assert.ok(isArtifactManifest(parsed));
  const manifest = parsed;

  assert.deepEqual(entries.filter((entry) => entry.endsWith(".wasm")), ["sensor-engine.abi1.wasm"]);
  assert.equal(manifest.abiVersion, 1);
  assert.equal(manifest.filename, "sensor-engine.abi1.wasm");
  assert.equal(manifest.byteLength, artifact.byteLength);
  assert.equal(manifest.sha256, createHash("sha256").update(artifact).digest("hex"));
  assert.deepEqual(manifest.exports, ["a", "d", "memory", "x"]);
  assert.equal(manifest.maximumInputBytes, 1024 * 1024);
  assert.equal(manifest.maximumOutputBytes, 1024 * 1024);
  assert.doesNotThrow(() => new WebAssembly.Module(artifact));
  const verified = await verifySensorArtifact();
  assert.deepEqual(verified, {
    filename: "sensor-engine.abi1.wasm",
    manifestFilename: "sensor-engine.manifest.json",
    byteLength: artifact.byteLength,
    sha256: manifest.sha256,
    exports: ["a", "d", "memory", "x"],
  });
});

test("keeps the provisional ABI 1 asset revalidated and serves it under a first-party WASM CSP", async () => {
  const config = await readFile(resolve(import.meta.dirname, "../../next.config.ts"), "utf8");
  const vercel = await readFile(resolve(import.meta.dirname, "../../vercel.json"), "utf8");
  assert.match(config, /source: "\/sensor\/sensor-engine\.abi1\.wasm"/);
  assert.match(config, /public, max-age=0, must-revalidate/);
  assert.match(vercel, /"source": "\/sensor\/sensor-engine\.abi1\.wasm"/);
  assert.match(vercel, /"public, max-age=0, must-revalidate"/);
  assert.match(config, /Content-Security-Policy/);
  assert.match(config, /script-src \$\{scriptSources\.join\(" "\)\}/);
  assert.match(config, /"'self'", "'unsafe-inline'", "'wasm-unsafe-eval'"/);
  assert.match(config, /"connect-src 'self'"/);
  assert.doesNotMatch(config, /https?:\/\/(?!opencgm\.vercel\.app|localhost)/);
  assert.doesNotMatch(config, /relay|private|third-party/i);
});

function isArtifactManifest(value: unknown): value is ArtifactManifest {
  if (typeof value !== "object" || value === null) return false;
  if (!("abiVersion" in value) || typeof value.abiVersion !== "number") return false;
  if (!("filename" in value) || typeof value.filename !== "string") return false;
  if (!("byteLength" in value) || typeof value.byteLength !== "number") return false;
  if (!("sha256" in value) || typeof value.sha256 !== "string") return false;
  if (!("exports" in value) || !Array.isArray(value.exports) || !value.exports.every((entry): entry is string => typeof entry === "string")) return false;
  if (!("toolchain" in value) || typeof value.toolchain !== "object" || value.toolchain === null) return false;
  if (!("licenses" in value) || !Array.isArray(value.licenses) || !value.licenses.every((entry): entry is string => typeof entry === "string")) return false;
  return true;
}
