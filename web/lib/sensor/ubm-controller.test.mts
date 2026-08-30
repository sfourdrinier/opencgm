import assert from "node:assert/strict";
import test from "node:test";
import { createSensorController, getWebSensorSupport, type SensorControllerDependencies } from "./ubm-controller";
import type { EngineAction, EngineCommand, ImportedSensorReading } from "./contracts";

function fakeManager() {
  const calls: string[] = [];
  const characteristic = (uuid: string) => ({
    subscribe: async () => { calls.push(`subscribe:${uuid}`); return { values: (async function* () {})(), remove: async () => undefined }; },
    write: async (value: Uint8Array, options?: { response?: string }) => { calls.push(`write:${uuid}:${options?.response ?? "automatic"}:${value[0]}`); return undefined; },
  });
  const gatt = { characteristic: (service: string, uuid: string) => characteristic(`${service}/${uuid}`) };
  return {
    calls,
    manager: {
      choose: async () => { calls.push("choose"); return { id: "peer", name: "sensor", rssi: null, reference: null, sources: [], lastAdvertisement: null }; },
      connect: async () => { calls.push("connect"); return { discover: async () => { calls.push("discover"); return gatt; }, disconnect: async () => { calls.push("disconnect"); return { state: "released", failures: [] }; } }; },
      destroy: async () => { calls.push("destroy"); return { state: "released", failures: [] }; },
    },
  };
}

const reading: ImportedSensorReading = {
  sensorId: "sensor-a", sensorSeconds: 1, atMs: 2_000, receivedAtMs: 2_100,
  mgdl: 101, reliable: true, algorithmState: 1, source: "live",
};

test("exposes browser capability state from secure context and navigator Bluetooth", () => {
  assert.deepEqual(getWebSensorSupport({ secureContext: true, bluetooth: true, origin: "https://opencgm.vercel.app" }), { state: "supported" });
  assert.deepEqual(getWebSensorSupport({ secureContext: false, bluetooth: true, origin: "http://localhost:3000" }), { state: "supported" });
  assert.deepEqual(getWebSensorSupport({ secureContext: false, bluetooth: true, origin: "http://opencgm.vercel.app" }), { state: "unsupported", reason: "secure-context-required" });
  assert.deepEqual(getWebSensorSupport({ secureContext: true, bluetooth: false }), { state: "unsupported", reason: "web-bluetooth-unavailable" });
});

test("keeps the chooser in the user-activation path and orders subscriptions before writes", async () => {
  const fake = fakeManager();
  const actions: EngineAction[] = [
    { kind: "subscribe", actionId: 1, channel: "authentication" },
    { kind: "write", actionId: 2, channel: "authentication", response: "required", bytes: new Uint8Array([1]), delayAfterMs: 0 },
    { kind: "reading", reading },
    { kind: "complete", completeness: "complete" },
  ];
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => ({ push: async (command: EngineCommand) => command.kind === "start" ? actions : [], stop: async () => undefined }),
    clock: () => 2_100,
    timer: { sleep: async () => undefined },
    entropy: (count) => new Uint8Array(count),
  };
  const controller = createSensorController(deps, { serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  const result = await controller.importSensor({ sensorName: "sensor-a", userActivation: true });

  assert.deepEqual(result.records, [reading]);
  assert.deepEqual(fake.calls.slice(0, 4), ["choose", "connect", "discover", "subscribe:service/auth"]);
  assert.equal(fake.calls[4], "write:service/auth:required:1");
});

test("truthfully rejects chooser use outside transient activation", async () => {
  const fake = fakeManager();
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => ({ push: async () => [], stop: async () => undefined }),
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });
  await assert.rejects(controller.importSensor({ sensorName: "sensor-a", userActivation: false }), /user activation/);
  assert.deepEqual(fake.calls, []);
});

test("retries after a failed action while preserving the partial history and cleaning each connection once", async () => {
  const fake = fakeManager();
  let starts = 0;
  let disconnects = 0;
  const first: EngineAction[] = [{ kind: "reading", reading }, { kind: "failure", category: "history-incomplete" }];
  const second: EngineAction[] = [{ kind: "complete", completeness: "complete" }];
  const manager = {
    ...fake.manager,
    connect: async () => ({
      discover: async () => ({ characteristic: () => ({ subscribe: async () => ({ values: (async function* () {})(), remove: async () => undefined }), write: async () => undefined }) }),
      disconnect: async () => { disconnects += 1; },
    }),
  };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({ push: async (command: EngineCommand) => command.kind === "start" ? (++starts === 1 ? first : second) : [], stop: async () => undefined }),
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 2, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  const result = await controller.importSensor({ sensorName: "sensor-a", userActivation: true });

  assert.deepEqual(result.records, [reading]);
  assert.equal(result.completeness, "complete");
  assert.equal(disconnects, 2);
});

test("fails truthfully when the engine requests unsupported browser OS pairing", async () => {
  const fake = fakeManager();
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => ({ push: async (command: EngineCommand) => command.kind === "start" ? [{ kind: "request-os-pair", actionId: 1 }] : [], stop: async () => undefined }),
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });
  await assert.rejects(controller.importSensor({ sensorName: "sensor-a", userActivation: true }), /does not support OS pairing/);
});
