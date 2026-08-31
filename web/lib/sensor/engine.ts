import {
  decodeEngineActions,
  encodeEngineCommand,
  ENGINE_ABI_MAX_BYTES,
  ENGINE_ABI_VERSION,
  type EngineAction,
  type EngineCommand,
} from "./contracts";

export const SENSOR_ENGINE_MANIFEST_PATH = "/sensor/sensor-engine.manifest.json";
const SENSOR_ENGINE_ARTIFACT_PREFIX = "/sensor/";
const SENSOR_ENGINE_ARTIFACT_FILENAME = "sensor-engine.abi1.wasm";
const REQUIRED_EXPORTS = ["a", "d", "memory", "x"] as const;
const REQUIRED_TOOL_RECEIPTS = ["rustc", "cargo", "wasmOpt", "wasmTools"] as const;

export type SensorEngineManifest = {
  readonly abiVersion: typeof ENGINE_ABI_VERSION;
  readonly filename: typeof SENSOR_ENGINE_ARTIFACT_FILENAME;
  readonly byteLength: number;
  readonly sha256: string;
  readonly exports: readonly (typeof REQUIRED_EXPORTS[number])[];
  readonly maximumInputBytes: typeof ENGINE_ABI_MAX_BYTES;
  readonly maximumOutputBytes: typeof ENGINE_ABI_MAX_BYTES;
  readonly toolchain: Readonly<Record<typeof REQUIRED_TOOL_RECEIPTS[number], string>>;
  readonly licenses: readonly string[];
};

export type SensorEngineRawExports = {
  readonly memory: WebAssembly.Memory;
  readonly a: (size: number) => number;
  readonly x: (pointer: number, size: number) => bigint;
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
  const wasmPath = `${SENSOR_ENGINE_ARTIFACT_PREFIX}${manifest.filename}`;
  const wasmBytes = await dependencies.fetchBytes(wasmPath);
  if (wasmBytes.byteLength !== manifest.byteLength) throw new Error("sensor engine byte length mismatch");
  const digest = await dependencies.sha256(wasmBytes);
  if (digest !== manifest.sha256) throw new Error("sensor engine digest mismatch");
  const exports = await dependencies.instantiate(wasmBytes);
  let stopped = false;

  const process = async (input: Uint8Array): Promise<Uint8Array> => {
    if (stopped) throw new Error("sensor engine is stopped");
    if (input.byteLength > manifest.maximumInputBytes) throw new Error("sensor engine input exceeds bound");
    const pointer = exports.a(input.byteLength);
    if (!Number.isSafeInteger(pointer) || pointer <= 0) throw new Error("sensor engine returned invalid input pointer");
    let output: { readonly pointer: number; readonly length: number } | null = null;
    try {
      copyIntoMemory(exports.memory, pointer, input);
      const packed = exports.x(pointer, input.byteLength);
      output = unpackResult(packed, manifest.maximumOutputBytes);
      const view = new Uint8Array(exports.memory.buffer);
      if (output.pointer > view.byteLength || output.length > view.byteLength - output.pointer) throw new Error("sensor engine output is out of bounds");
      return view.slice(output.pointer, output.pointer + output.length);
    } finally {
      if (output !== null) exports.d(output.pointer, output.length);
      exports.d(pointer, input.byteLength);
    }
  };

  return {
    process,
    push: async command => decodeEngineActions(await process(encodeEngineCommand(command))),
    stop: async () => { stopped = true; },
    manifest,
  };
}

function parseManifest(value: unknown): SensorEngineManifest {
  if (!isRecord(value) || value.abiVersion !== ENGINE_ABI_VERSION || value.filename !== SENSOR_ENGINE_ARTIFACT_FILENAME) throw new Error("invalid sensor engine manifest");
  if (typeof value.byteLength !== "number" || !Number.isSafeInteger(value.byteLength) || value.byteLength < 1 || value.byteLength > ENGINE_ABI_MAX_BYTES) throw new Error("invalid sensor engine byte length");
  if (typeof value.sha256 !== "string" || !/^[0-9a-f]{64}$/u.test(value.sha256)) throw new Error("invalid sensor engine digest");
  if (!Array.isArray(value.exports) || !equalStrings(value.exports, REQUIRED_EXPORTS)) throw new Error("invalid sensor engine exports");
  if (value.maximumInputBytes !== ENGINE_ABI_MAX_BYTES || value.maximumOutputBytes !== ENGINE_ABI_MAX_BYTES) throw new Error("invalid sensor engine bounds");
  const toolchain = value.toolchain;
  if (!isRecord(toolchain) || REQUIRED_TOOL_RECEIPTS.some(key => typeof toolchain[key] !== "string" || toolchain[key] === "")) throw new Error("invalid sensor engine tool receipt");
  const licenses = value.licenses;
  if (!Array.isArray(licenses) || licenses.length === 0 || licenses.some(license => typeof license !== "string" || license === "")) throw new Error("invalid sensor engine licenses");
  return {
    abiVersion: ENGINE_ABI_VERSION,
    filename: SENSOR_ENGINE_ARTIFACT_FILENAME,
    byteLength: value.byteLength,
    sha256: value.sha256,
    exports: [...REQUIRED_EXPORTS],
    maximumInputBytes: ENGINE_ABI_MAX_BYTES,
    maximumOutputBytes: ENGINE_ABI_MAX_BYTES,
    toolchain: toolchain as SensorEngineManifest["toolchain"],
    licenses: licenses as string[],
  };
}

function unpackResult(value: unknown, maximum: number): { readonly pointer: number; readonly length: number } {
  if (typeof value !== "bigint" || value < 0n || value > 0xffff_ffff_ffff_ffffn) throw new Error("sensor engine returned invalid packed output");
  const pointer = Number(value & 0xffff_ffffn);
  const length = Number(value >> 32n);
  if (!Number.isSafeInteger(pointer) || pointer <= 0 || !Number.isSafeInteger(length) || length <= 0 || length > maximum) throw new Error("sensor engine returned invalid packed output");
  return { pointer, length };
}

function equalStrings(actual: readonly unknown[], expected: readonly string[]): actual is readonly string[] {
  return actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function bytesForDigest(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function copyIntoMemory(memory: WebAssembly.Memory, pointer: number, bytes: Uint8Array): void {
  const view = new Uint8Array(memory.buffer);
  if (pointer > view.byteLength || bytes.byteLength > ENGINE_ABI_MAX_BYTES || pointer > view.byteLength - bytes.byteLength) throw new Error("sensor engine input allocation is out of bounds");
  view.set(bytes, pointer);
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
      if (typeof result !== "bigint") throw new Error("sensor engine executor returned a non-bigint");
      return result;
    },
    d: (pointer, size) => { Reflect.apply(deallocator, undefined, [pointer, size]); },
  };
}
