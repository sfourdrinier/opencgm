import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadSensorEngine, type SensorEngineLoaderDependencies } from "./engine";

function wasmFixture() {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const input = 32;
  const output = 128;
  const view = new Uint8Array(memory.buffer);
  view.set([9, 8, 7, 6], output);
  const freed: Array<[number, number]> = [];
  return {
    memory,
    freed,
    exports: {
      memory,
      a: (size: number) => { assert.equal(size, 3); return input; },
      x: (pointer: number, size: number) => { assert.equal(pointer, input); assert.equal(size, 3); return (4n << 32n) | BigInt(output); },
      d: (pointer: number, size: number) => { freed.push([pointer, size]); },
    },
  };
}

function manifest(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    abiVersion: 1,
    filename: "sensor-engine.abi1.wasm",
    byteLength: 3,
    sha256: "0".repeat(64),
    exports: ["a", "d", "memory", "x"],
    maximumInputBytes: 1024 * 1024,
    maximumOutputBytes: 1024 * 1024,
    toolchain: { rustc: "rustc", cargo: "cargo", wasmOpt: "wasm-opt", wasmTools: "wasm-tools" },
    licenses: ["MIT OR Apache-2.0"],
    ...overrides,
  };
}

test("loads the checked-in manifest path and shape before instantiating", async () => {
  const manifestText = await readFile(resolve(import.meta.dirname, "../../public/sensor/sensor-engine.manifest.json"), "utf8");
  const manifest = JSON.parse(manifestText) as Record<string, unknown>;
  manifest.byteLength = 3;
  manifest.sha256 = "0".repeat(64);
  const fixture = wasmFixture();
  const bytes = new Uint8Array([1, 2, 3]);
  const calls: string[] = [];
  const deps: SensorEngineLoaderDependencies = {
    fetch: async (path) => {
      assert.equal(path, "/sensor/sensor-engine.manifest.json");
      return new Response(JSON.stringify(manifest));
    },
    fetchBytes: async (path) => {
      assert.equal(path, "/sensor/sensor-engine.abi1.wasm");
      return bytes;
    },
    sha256: async (value) => { assert.deepEqual(value, bytes); calls.push("hash"); return "0".repeat(64); },
    instantiate: async () => { calls.push("instantiate"); return fixture.exports; },
  };

  const engine = await loadSensorEngine(deps);
  assert.deepEqual(await engine.process(new Uint8Array([4, 5, 6])), new Uint8Array([9, 8, 7, 6]));
  assert.deepEqual(calls, ["hash", "instantiate"]);
  assert.deepEqual(fixture.freed, [[128, 4], [32, 3]]);
});

test("dispatches start and stop through the checked-in ABI 1 artifact", async () => {
  const sensorRoot = resolve(import.meta.dirname, "../../public/sensor");
  const manifestText = await readFile(resolve(sensorRoot, "sensor-engine.manifest.json"), "utf8");
  const wasmBytes = new Uint8Array(await readFile(resolve(sensorRoot, "sensor-engine.abi1.wasm")));
  const freed: Array<[number, number]> = [];
  const deps: SensorEngineLoaderDependencies = {
    fetch: async path => {
      assert.equal(path, "/sensor/sensor-engine.manifest.json");
      return new Response(manifestText);
    },
    fetchBytes: async path => {
      assert.equal(path, "/sensor/sensor-engine.abi1.wasm");
      return wasmBytes;
    },
    sha256: async value => createHash("sha256").update(value).digest("hex"),
    instantiate: async value => {
      const { instance } = await WebAssembly.instantiate(value, {});
      const raw = instance.exports;
      return {
        memory: raw.memory as WebAssembly.Memory,
        a: raw.a as (size: number) => number,
        x: raw.x as (pointer: number, size: number) => bigint,
        d: (pointer: number, size: number) => { freed.push([pointer, size]); (raw.d as (pointer: number, size: number) => void)(pointer, size); },
      };
    },
  };

  const engine = await loadSensorEngine(deps);
  assert.deepEqual(await engine.push({ kind: "start", nowMs: 1, sensorName: "sensor", credential: null, pairingCode: null, certificateBundle: null }), [
    { kind: "subscribe", actionId: 1, channel: "authentication" },
  ]);
  assert.deepEqual(await engine.push({ kind: "stop" }), [{ kind: "failure", category: "cancelled" }]);
  assert.equal(freed.length, 4);
  assert.ok(freed.every(([pointer, length]) => pointer > 0 && length > 0));
});

test("frees the input allocation when execution throws", async () => {
  const fixture = wasmFixture();
  fixture.exports.x = () => { throw new Error("engine failed"); };
  const deps: SensorEngineLoaderDependencies = {
    fetch: async () => new Response(JSON.stringify(manifest())),
    fetchBytes: async () => new Uint8Array([1, 2, 3]),
    sha256: async () => "0".repeat(64),
    instantiate: async () => fixture.exports,
  };
  const engine = await loadSensorEngine(deps);
  await assert.rejects(engine.process(new Uint8Array([4, 5, 6])), /engine failed/);
  assert.deepEqual(fixture.freed, [[32, 3]]);
});

test("rejects a digest mismatch before instantiating untrusted bytes", async () => {
  let instantiated = false;
  const deps: SensorEngineLoaderDependencies = {
    fetch: async () => new Response(JSON.stringify(manifest({ byteLength: 1, sha256: "0".repeat(63) + "1" }))),
    fetchBytes: async () => new Uint8Array([1]),
    sha256: async () => "actual",
    instantiate: async () => { instantiated = true; return wasmFixture().exports; },
  };
  await assert.rejects(loadSensorEngine(deps), /digest mismatch/);
  assert.equal(instantiated, false);
});

test("rejects malformed packed output and frees the input allocation", async () => {
  const fixture = wasmFixture();
  fixture.exports.x = () => 0n;
  const deps: SensorEngineLoaderDependencies = {
    fetch: async () => new Response(JSON.stringify(manifest())),
    fetchBytes: async () => new Uint8Array([1, 2, 3]),
    sha256: async () => "0".repeat(64),
    instantiate: async () => fixture.exports,
  };
  const engine = await loadSensorEngine(deps);
  await assert.rejects(engine.process(new Uint8Array([4, 5, 6])), /packed output/);
  assert.deepEqual(fixture.freed, [[32, 3]]);
});
