// web/scripts/verify-sensor-artifact.ts

import { createHash } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";

const ABI_VERSION = 1;
const ARTIFACT_FILENAME = "sensor-engine.abi1.wasm";
const MANIFEST_FILENAME = "sensor-engine.manifest.json";
const ARTIFACT_MAX_BYTES = 1024 * 1024;
const REQUIRED_EXPORTS = Object.freeze(["a", "d", "memory", "x"]);

const forbiddenPatterns = Object.freeze([
  /dexcom/i,
  /\bcgm\b/i,
  /\bg7\b/i,
  /glucose/i,
  /sensor/i,
  /jpake/i,
  /certificate/i,
  /credential/i,
  /backfill/i,
  /authentication/i,
  /protocol/i,
  /diagnostic/i,
  /pairing/i,
  /xdrip/i,
  /juggluco/i,
  /private[_ -]?key/i,
  /secret/i,
  /sourcemappingurl/i,
  /\.map(?:$|[^a-z])/i,
  /(?:^|[/\\])(home|users|private|workspace|target)(?:[/\\]|$)/i,
  /bun-mono-webdexcom/i,
  /packages[/\\]cgm-dexcom/i,
  /rust[/\\]src[/\\](?:protocol|crypto|claim|certificate|session|history|lifecycle)[^/\\]*\.rs/i,
  /BEGIN (?:RSA |EC )?PRIVATE KEY/i,
]);

export type SensorArtifactVerification = {
  readonly filename: typeof ARTIFACT_FILENAME;
  readonly manifestFilename: typeof MANIFEST_FILENAME;
  readonly byteLength: number;
  readonly sha256: string;
  readonly exports: readonly string[];
};

type ArtifactManifest = {
  readonly abiVersion: typeof ABI_VERSION;
  readonly filename: typeof ARTIFACT_FILENAME;
  readonly byteLength: number;
  readonly sha256: string;
  readonly exports: readonly string[];
  readonly toolchain: { readonly [key: string]: string };
  readonly licenses: readonly string[];
};

type Leb128 = { readonly value: number; readonly next: number };

export type SensorArtifactVerificationOptions = {
  readonly artifactPath?: string;
  readonly manifestPath?: string;
  readonly directoryPath?: string;
};

const defaultDirectory = resolve(import.meta.dirname, "../public/sensor");

function fail(message: string): never {
  throw new Error(`Sensor artifact verification failed: ${message}`);
}

function equalStrings(actual: readonly string[], expected: readonly string[]): boolean {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

function readUnsignedLeb128(bytes: Uint8Array, offset: number): Leb128 {
  let value = 0;
  let shift = 0;
  let cursor = offset;
  while (cursor < bytes.length && shift < 35) {
    const byte = bytes[cursor];
    if (byte === undefined) break;
    cursor += 1;
    value += (byte & 0x7f) * 2 ** shift;
    if ((byte & 0x80) === 0) return { value, next: cursor };
    shift += 7;
  }
  fail("malformed WebAssembly section length");
}

function customSectionNames(bytes: Uint8Array): readonly string[] {
  if (bytes.length < 8 || bytes[0] !== 0 || bytes[1] !== 0x61 || bytes[2] !== 0x73 || bytes[3] !== 0x6d) {
    fail("invalid WebAssembly header");
  }
  const names: string[] = [];
  let offset = 8;
  while (offset < bytes.length) {
    const sectionId = bytes[offset];
    if (sectionId === undefined) fail("missing section identifier");
    offset += 1;
    const section = readUnsignedLeb128(bytes, offset);
    offset = section.next;
    const end = offset + section.value;
    if (end > bytes.length) fail("section extends beyond module");
    if (sectionId === 0) {
      const nameLength = readUnsignedLeb128(bytes, offset);
      const nameEnd = nameLength.next + nameLength.value;
      if (nameEnd > end) fail("malformed custom section name");
      names.push(new TextDecoder("utf-8", { fatal: false }).decode(bytes.slice(nameLength.next, nameEnd)));
    }
    offset = end;
  }
  return names;
}

function printableStrings(bytes: Uint8Array): readonly string[] {
  const strings: string[] = [];
  let start = -1;
  for (let index = 0; index <= bytes.length; index += 1) {
    const byte = bytes[index] ?? 0;
    const printable = byte >= 0x20 && byte <= 0x7e;
    if (printable && start < 0) start = index;
    if ((!printable || index === bytes.length) && start >= 0) {
      if (index - start >= 4) strings.push(Buffer.from(bytes.slice(start, index)).toString("ascii"));
      start = -1;
    }
  }
  return strings;
}

function isManifest(value: unknown): value is ArtifactManifest {
  if (typeof value !== "object" || value === null) return false;
  if (!("abiVersion" in value) || value.abiVersion !== ABI_VERSION) return false;
  if (!("filename" in value) || value.filename !== ARTIFACT_FILENAME) return false;
  if (!("byteLength" in value) || typeof value.byteLength !== "number") return false;
  if (!("sha256" in value) || typeof value.sha256 !== "string") return false;
  if (!("exports" in value) || !Array.isArray(value.exports) || !value.exports.every((entry): entry is string => typeof entry === "string")) return false;
  if (!("toolchain" in value) || !isStringRecord(value.toolchain)) return false;
  if (!("licenses" in value) || !Array.isArray(value.licenses) || !value.licenses.every((entry): entry is string => typeof entry === "string")) return false;
  return true;
}

function isStringRecord(value: unknown): value is { readonly [key: string]: string } {
  return typeof value === "object" && value !== null && Object.values(value).every((entry) => typeof entry === "string");
}

function parseManifest(text: string): ArtifactManifest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    const reason = error instanceof Error ? error.message : "invalid JSON";
    fail(`manifest is not valid JSON: ${reason}`);
  }
  if (!isManifest(parsed)) fail("manifest shape or ABI is invalid");
  if (!Number.isSafeInteger(parsed.byteLength) || parsed.byteLength < 1 || parsed.byteLength > ARTIFACT_MAX_BYTES) {
    fail("manifest byte length is outside the artifact cap");
  }
  if (!/^[0-9a-f]{64}$/u.test(parsed.sha256)) fail("manifest SHA-256 is not lowercase hex");
  if (!equalStrings([...parsed.exports].sort(), [...REQUIRED_EXPORTS].sort())) fail("manifest exports are not exact");
  for (const key of ["rustc", "cargo", "wasmOpt", "wasmTools"]) {
    if (parsed.toolchain[key] === undefined || parsed.toolchain[key].length === 0) fail(`manifest tool receipt ${key} is missing`);
  }
  if (parsed.licenses.length === 0) fail("manifest licence inventory is missing");
  return parsed;
}

function verifyModule(bytes: Uint8Array): readonly string[] {
  let wasmModule: WebAssembly.Module;
  try {
    wasmModule = new WebAssembly.Module(bytesForWasm(bytes));
  } catch (error) {
    const reason = error instanceof Error ? error.message : "invalid WebAssembly module";
    fail(`module validation failed: ${reason}`);
  }
  const exports = WebAssembly.Module.exports(wasmModule);
  const names = exports.map((entry) => entry.name).sort();
  if (!equalStrings(names, [...REQUIRED_EXPORTS].sort())) fail(`module exports ${names.join(", ")} instead of the exact ABI surface`);
  for (const entry of exports) {
    const expectedKind = entry.name === "memory" ? "memory" : "function";
    if (entry.kind !== expectedKind) fail(`export ${entry.name} has kind ${entry.kind}, expected ${expectedKind}`);
  }
  return names;
}

function bytesForWasm(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

export async function verifySensorArtifact(options: SensorArtifactVerificationOptions = {}): Promise<SensorArtifactVerification> {
  const directoryPath = options.directoryPath ?? defaultDirectory;
  const artifactPath = options.artifactPath ?? resolve(directoryPath, ARTIFACT_FILENAME);
  const manifestPath = options.manifestPath ?? resolve(directoryPath, MANIFEST_FILENAME);
  let bytes: Buffer;
  try {
    bytes = await readFile(artifactPath);
  } catch (error) {
    const reason = error instanceof Error ? error.message : "unable to read artifact";
    fail(`cannot read artifact: ${reason}`);
  }
  let manifestText: string;
  try {
    manifestText = await readFile(manifestPath, "utf8");
  } catch (error) {
    const reason = error instanceof Error ? error.message : "unable to read manifest";
    fail(`cannot read manifest: ${reason}`);
  }
  const manifest = parseManifest(manifestText);
  if (basename(artifactPath) !== ARTIFACT_FILENAME) fail(`artifact filename must be ${ARTIFACT_FILENAME}`);
  if (basename(manifestPath) !== MANIFEST_FILENAME) fail(`manifest filename must be ${MANIFEST_FILENAME}`);
  if (bytes.byteLength > ARTIFACT_MAX_BYTES) fail(`artifact is ${bytes.byteLength} bytes, above ${ARTIFACT_MAX_BYTES}-byte cap`);
  if (manifest.byteLength !== bytes.byteLength) fail("manifest byte length does not match artifact");
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 !== manifest.sha256) fail("manifest SHA-256 does not match artifact");
  const exports = verifyModule(bytes);
  const sections = customSectionNames(bytes);
  const forbiddenSections = sections.filter((name) => /^(?:name|producers|sourcemappingurl|\.debug_|reloc\.)$/iu.test(name) || /(?:source.?map|debug|producer)/iu.test(name));
  if (forbiddenSections.length > 0) fail(`forbidden custom sections: ${forbiddenSections.join(", ")}`);
  const matches: string[] = [];
  for (const value of printableStrings(bytes)) {
    for (const pattern of forbiddenPatterns) {
      if (pattern.test(value)) {
        matches.push(value);
        break;
      }
    }
  }
  if (matches.length > 0) fail(`forbidden printable strings: ${matches.slice(0, 8).join(", ")}`);
  const entries = await readdir(directoryPath);
  const wasmEntries = entries.filter((entry) => entry.endsWith(".wasm"));
  if (!equalStrings(wasmEntries, [ARTIFACT_FILENAME])) fail(`sensor directory must contain only ${ARTIFACT_FILENAME} as a WebAssembly file`);
  return { filename: ARTIFACT_FILENAME, manifestFilename: MANIFEST_FILENAME, byteLength: bytes.byteLength, sha256, exports };
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === resolve(import.meta.filename)) {
  verifySensorArtifact()
    .then((result) => {
      process.stdout.write(`sensor artifact verified: ${result.byteLength} bytes, sha256=${result.sha256}, exports=${result.exports.join(",")}\n`);
    })
    .catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      process.stderr.write(`${message}\n`);
      process.exitCode = 1;
    });
}
