import { ENGINE_ABI_VERSION, type EngineAction, type EngineCommand } from "./contracts";

export const SENSOR_ENGINE_MANIFEST_PATH = "/sensor/engine.manifest.json";

export type SensorEngineManifest = {
  readonly abiVersion: number;
  readonly wasmPath: string;
  readonly sha256: string;
  readonly maximumInputBytes: number;
  readonly maximumOutputBytes: number;
};

export type SensorEngineRawExports = {
  readonly memory: WebAssembly.Memory;
  readonly a: (size: number) => number;
  readonly x: (pointer: number, size: number) => number;
  readonly d: (pointer: number, size: number) => void;
};

export type SensorEngineLoaderDependencies = {
  readonly fetch: (path: string) => Promise<Response>;
  readonly fetchBytes: (path: string) => Promise<Uint8Array>;
  readonly sha256: (bytes: Uint8Array) => Promise<string>;
  readonly instantiate: (bytes: Uint8Array) => Promise<SensorEngineRawExports>;
};

export type SensorEngine = {
  readonly process: (bytes: Uint8Array) => Promise<Uint8Array>;
  readonly push: (command: EngineCommand) => Promise<readonly EngineAction[]>;
  readonly stop: () => Promise<void>;
  readonly manifest: SensorEngineManifest;
};

function defaultDependencies(): SensorEngineLoaderDependencies {
  return {
    fetch: async path => fetch(path),
    fetchBytes: async path => {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`sensor engine fetch failed: ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    },
    sha256: async bytes => {
      const digest = await crypto.subtle.digest("SHA-256", bytesForDigest(bytes));
      return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, "0")).join("");
    },
    instantiate: async bytes => {
      const instance = await WebAssembly.instantiate(bytes, {});
      return validateExports(instance.exports);
    },
  };
}

export async function loadSensorEngine(
  dependencies: SensorEngineLoaderDependencies = defaultDependencies(),
): Promise<SensorEngine> {
  const manifestResponse = await dependencies.fetch(SENSOR_ENGINE_MANIFEST_PATH);
  if (!manifestResponse.ok) throw new Error(`sensor engine manifest fetch failed: ${manifestResponse.status}`);
  const manifest = parseManifest(await manifestResponse.json());
  const wasmBytes = await dependencies.fetchBytes(manifest.wasmPath);
  const digest = await dependencies.sha256(wasmBytes);
  if (digest !== manifest.sha256) throw new Error("sensor engine digest mismatch");
  const exports = await dependencies.instantiate(wasmBytes);
  let stopped = false;

  const process = async (input: Uint8Array): Promise<Uint8Array> => {
    if (stopped) throw new Error("sensor engine is stopped");
    if (input.byteLength > manifest.maximumInputBytes) throw new Error("sensor engine input exceeds bound");
    const pointer = exports.a(input.byteLength);
    if (!Number.isSafeInteger(pointer) || pointer < 0) throw new Error("sensor engine returned invalid input pointer");
    let outputPointer: number | null = null;
    let outputLength = 0;
    try {
      copyIntoMemory(exports.memory, pointer, input, manifest.maximumInputBytes);
      outputPointer = exports.x(pointer, input.byteLength);
      if (!Number.isSafeInteger(outputPointer) || outputPointer < 0) throw new Error("sensor engine returned invalid output pointer");
      outputLength = readLength(exports.memory, outputPointer, manifest.maximumOutputBytes);
      return readOutput(exports.memory, outputPointer, outputLength);
    } finally {
      if (outputPointer !== null) {
        exports.d(outputPointer, outputLength);
      }
      exports.d(pointer, input.byteLength);
    }
  };

  return {
    process,
    push: async command => {
      const encoded = new TextEncoder().encode(JSON.stringify(command, (_key, value: unknown) => value instanceof Uint8Array ? [...value] : value));
      const output = await process(encoded);
      const decoded: unknown = JSON.parse(new TextDecoder().decode(output));
      if (!Array.isArray(decoded)) throw new Error("sensor engine returned a non-action list");
      return decoded.filter(isEngineAction);
    },
    stop: async () => { stopped = true; },
    manifest,
  };
}

function parseManifest(value: unknown): SensorEngineManifest {
  if (typeof value !== "object" || value === null) throw new Error("invalid sensor engine manifest");
  if (!isRecord(value)) throw new Error("invalid sensor engine manifest");
  const candidate = value;
  if (candidate.abiVersion !== ENGINE_ABI_VERSION || typeof candidate.wasmPath !== "string" || !candidate.wasmPath.startsWith("/sensor/")) throw new Error("invalid sensor engine manifest");
  if (typeof candidate.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(candidate.sha256)) throw new Error("invalid sensor engine digest");
  if (!isPositiveFiniteNumber(candidate.maximumInputBytes) || !isPositiveFiniteNumber(candidate.maximumOutputBytes)) throw new Error("invalid sensor engine bounds");
  return { abiVersion: ENGINE_ABI_VERSION, wasmPath: candidate.wasmPath, sha256: candidate.sha256, maximumInputBytes: candidate.maximumInputBytes, maximumOutputBytes: candidate.maximumOutputBytes };
}

function isPositiveFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function isRecord(value: unknown): value is { readonly [key: string]: unknown } {
  return typeof value === "object" && value !== null;
}

function bytesForDigest(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function copyIntoMemory(memory: WebAssembly.Memory, pointer: number, bytes: Uint8Array, maximum: number): void {
  const view = new Uint8Array(memory.buffer);
  if (pointer > view.byteLength || bytes.byteLength > maximum || pointer + bytes.byteLength > view.byteLength) throw new Error("sensor engine input allocation is out of bounds");
  view.set(bytes, pointer);
}

function readLength(memory: WebAssembly.Memory, pointer: number, maximum: number): number {
  const view = new DataView(memory.buffer);
  if (pointer > view.byteLength - 4) throw new Error("sensor engine output header is out of bounds");
  const length = view.getUint32(pointer, true);
  if (length > maximum || pointer + 4 + length > view.byteLength) throw new Error("sensor engine output exceeds bound");
  return length;
}

function readOutput(memory: WebAssembly.Memory, pointer: number, length: number): Uint8Array {
  return new Uint8Array(memory.buffer.slice(pointer + 4, pointer + 4 + length));
}

function validateExports(exports: WebAssembly.Exports): SensorEngineRawExports {
  const memory = exports.memory;
  const allocator = exports.a;
  const execute = exports.x;
  const deallocator = exports.d;
  if (!(memory instanceof WebAssembly.Memory) || typeof allocator !== "function" || typeof execute !== "function" || typeof deallocator !== "function") throw new Error("sensor engine ABI exports must be memory,a,x,d");
  return {
    memory,
    a: size => {
      const result: unknown = Reflect.apply(allocator, undefined, [size]);
      if (typeof result !== "number") throw new Error("sensor engine allocator returned a non-number");
      return result;
    },
    x: (pointer, size) => {
      const result: unknown = Reflect.apply(execute, undefined, [pointer, size]);
      if (typeof result !== "number") throw new Error("sensor engine executor returned a non-number");
      return result;
    },
    d: (pointer, size) => { Reflect.apply(deallocator, undefined, [pointer, size]); },
  };
}

function isEngineAction(value: unknown): value is EngineAction {
  return typeof value === "object" && value !== null && "kind" in value && typeof value.kind === "string";
}
