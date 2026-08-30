import assert from "node:assert/strict";
import test from "node:test";
import { loadSensorEngine, type SensorEngineLoaderDependencies } from "./engine";

function wasmFixture() {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const input = 32;
  const output = 128;
  const view = new Uint8Array(memory.buffer);
  view.set([4, 0, 0, 0, 9, 8, 7, 6], output);
  const freed: Array<[number, number]> = [];
  return {
    memory,
    freed,
    exports: {
      memory,
      a: (size: number) => { assert.equal(size, 3); return input; },
      x: (pointer: number, size: number) => { assert.equal(pointer, input); assert.equal(size, 3); return output; },
      d: (pointer: number, size: number) => { freed.push([pointer, size]); },
    },
  };
}

test("verifies the checked-in manifest digest before instantiation and copies bounded output", async () => {
  const fixture = wasmFixture();
  const bytes = new Uint8Array([1, 2, 3]);
  const calls: string[] = [];
  const deps: SensorEngineLoaderDependencies = {
    fetch: async (path) => {
      assert.equal(path, "/sensor/engine.manifest.json");
      return new Response(JSON.stringify({ abiVersion: 1, wasmPath: "/sensor/engine.wasm", sha256: "0".repeat(64), maximumInputBytes: 16, maximumOutputBytes: 16 }));
    },
    fetchBytes: async (path) => {
      assert.equal(path, "/sensor/engine.wasm");
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

test("frees the input allocation when execution throws", async () => {
  const fixture = wasmFixture();
  fixture.exports.x = () => { throw new Error("engine failed"); };
  const deps: SensorEngineLoaderDependencies = {
    fetch: async () => new Response(JSON.stringify({ abiVersion: 1, wasmPath: "/sensor/engine.wasm", sha256: "0".repeat(64), maximumInputBytes: 16, maximumOutputBytes: 16 })),
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
    fetch: async () => new Response(JSON.stringify({ abiVersion: 1, wasmPath: "/sensor/engine.wasm", sha256: "0".repeat(63) + "1", maximumInputBytes: 16, maximumOutputBytes: 16 })),
    fetchBytes: async () => new Uint8Array([1]),
    sha256: async () => "actual",
    instantiate: async () => { instantiated = true; return wasmFixture().exports; },
  };
  await assert.rejects(loadSensorEngine(deps), /digest mismatch/);
  assert.equal(instantiated, false);
});

test("frees a malformed output allocation even when its length header is invalid", async () => {
  const fixture = wasmFixture();
  new DataView(fixture.memory.buffer).setUint32(128, 999, true);
  const deps: SensorEngineLoaderDependencies = {
    fetch: async () => new Response(JSON.stringify({ abiVersion: 1, wasmPath: "/sensor/engine.wasm", sha256: "0".repeat(64), maximumInputBytes: 16, maximumOutputBytes: 16 })),
    fetchBytes: async () => new Uint8Array([1, 2, 3]),
    sha256: async () => "0".repeat(64),
    instantiate: async () => fixture.exports,
  };
  const engine = await loadSensorEngine(deps);
  await assert.rejects(engine.process(new Uint8Array([4, 5, 6])), /output/);
  assert.deepEqual(fixture.freed, [[128, 0], [32, 3]]);
});
