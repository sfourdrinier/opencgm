import assert from "node:assert/strict";
import test from "node:test";
import { createSensorController, getWebSensorSupport, type SensorChooserRequest, type SensorControllerDependencies } from "./ubm-controller";
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

test("passes advertisement filter and actual GATT service as optional chooser service", async () => {
  let chooserRequest: unknown;
  const fake = fakeManager();
  const manager = {
    ...fake.manager,
    choose: async (request: SensorChooserRequest) => { chooserRequest = request; return { id: "peer" }; },
  };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({ push: async () => [], stop: async () => undefined }),
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { chooserServiceUuid: "advertised", serviceUuid: "gatt-service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await controller.importSensor({ sensorName: "sensor", userActivation: true, pairingCode: "1234" });

  assert.deepEqual(chooserRequest, { filters: [{ serviceUuids: ["advertised"] }], optionalServices: ["gatt-service"] });
});

test("passes a bounded connection deadline through the manager contract", async () => {
  let connectOptions: unknown;
  const fake = fakeManager();
  const manager = {
    ...fake.manager,
    connect: async (peer: { id: string }, options?: { readonly timeoutMs: number }) => {
      connectOptions = options;
      return fake.manager.connect(peer);
    },
  };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({ push: async () => [], stop: async () => undefined }),
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { connectTimeoutMs: 42, maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await controller.importSensor({ sensorName: "sensor", userActivation: true });

  assert.deepEqual(connectOptions, { timeoutMs: 42 });
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

test("opens the chooser before engine loading can consume transient activation", async () => {
  let activationAvailable = true;
  let chooseCalls = 0;
  const fake = fakeManager();
  const deps: SensorControllerDependencies = {
    createManager: async () => ({
      ...fake.manager,
      choose: async () => {
        chooseCalls += 1;
        if (!activationAvailable) throw new Error("activation expired");
        return { id: "peer" };
      },
    }),
    createEngine: async () => {
      activationAvailable = false;
      return { push: async () => [], stop: async () => undefined };
    },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await controller.importSensor({ sensorName: "sensor", userActivation: true });

  assert.equal(chooseCalls, 1);
});

test("does not load or leak an engine when chooser cancellation ends the run", async () => {
  let engineLoads = 0;
  let managerDestroyed = 0;
  const deps: SensorControllerDependencies = {
    createManager: async () => ({
      choose: async () => { throw new Error("chooser cancelled"); },
      connect: async () => { throw new Error("connect should not run"); },
      destroy: async () => { managerDestroyed += 1; },
    }),
    createEngine: async () => {
      engineLoads += 1;
      return { push: async () => [], stop: async () => undefined };
    },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await assert.rejects(controller.importSensor({ sensorName: "sensor", userActivation: true }), /chooser cancelled/);

  assert.equal(engineLoads, 0);
  assert.equal(managerDestroyed, 1);
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

test("creates and stops a fresh engine for each protocol attempt", async () => {
  let engineCount = 0;
  const stoppedEngines: number[] = [];
  const fake = fakeManager();
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => {
      const engineId = ++engineCount;
      return {
        push: async (command: EngineCommand) => command.kind === "start"
          ? engineId === 1 ? [{ kind: "failure", category: "history-incomplete" }] : [{ kind: "complete", completeness: "complete" }]
          : [],
        stop: async () => { stoppedEngines.push(engineId); },
      };
    },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 2, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  const result = await controller.importSensor({ sensorName: "sensor", userActivation: true });

  assert.equal(result.completeness, "complete");
  assert.equal(engineCount, 2);
  assert.deepEqual(stoppedEngines, [1, 2]);
});

test("waits for asynchronous notification completion before cleaning up", async () => {
  let releaseFrame: (() => void) | undefined;
  const frameReady = new Promise<void>(resolve => { releaseFrame = resolve; });
  let settled = false;
  let engineStopped = 0;
  const fake = fakeManager();
  const manager = {
    ...fake.manager,
    connect: async () => ({
      discover: async () => ({
        characteristic: () => ({
          subscribe: async () => ({ values: (async function* () { await frameReady; yield { value: new Uint8Array([1]) }; })(), remove: async () => undefined }),
          write: async () => undefined,
        }),
      }),
      disconnect: async () => undefined,
    }),
  };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => {
        if (command.kind === "start") return [{ kind: "subscribe", actionId: 1, channel: "authentication" }];
        if (command.kind === "frame") return [{ kind: "complete", completeness: "complete" }];
        return [];
      },
      stop: async () => { engineStopped += 1; },
    }),
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  const importing = controller.importSensor({ sensorName: "sensor", userActivation: true });
  importing.then(() => { settled = true; });
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(settled, false);
  assert.equal(engineStopped, 0);
  releaseFrame?.();
  await importing;
  assert.equal(settled, true);
  assert.equal(engineStopped, 1);
});

test("delivers readings incrementally before a later notification failure", async () => {
  const delivered: ImportedSensorReading[] = [];
  const fake = fakeManager();
  const manager = {
    ...fake.manager,
    connect: async () => ({
      discover: async () => ({ characteristic: () => ({
        subscribe: async () => ({ values: (async function* () { yield { value: new Uint8Array([1]) }; throw new Error("stream lost"); })(), remove: async () => undefined }),
        write: async () => undefined,
      }) }),
      disconnect: async () => undefined,
    }),
  };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => {
        if (command.kind === "start") return [{ kind: "subscribe", actionId: 1, channel: "authentication" }];
        if (command.kind === "frame") return [{ kind: "reading", reading }];
        return [];
      },
      stop: async () => undefined,
    }),
    onReading: async current => { delivered.push(current); },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  const result = await controller.importSensor({ sensorName: "sensor", userActivation: true });

  assert.deepEqual(delivered, [reading]);
  assert.deepEqual(result.records, [reading]);
  assert.equal(result.completeness, "partial");
});

test("fails closed when incremental reading persistence fails", async () => {
  const fake = fakeManager();
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => command.kind === "start" ? [{ kind: "reading", reading }] : [],
      stop: async () => undefined,
    }),
    onReading: async () => { throw new Error("archive unavailable"); },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await assert.rejects(controller.importSensor({ sensorName: "sensor", userActivation: true }), /archive unavailable/);
});

test("rejects and sends one terminal command when the connection link is lost", async () => {
  let releaseLoss: (() => void) | undefined;
  const lossReady = new Promise<void>(resolve => { releaseLoss = resolve; });
  const notificationWaiting = new Promise<void>(() => undefined);
  const terminalCommands: EngineCommand[] = [];
  const fake = fakeManager();
  const manager = {
    ...fake.manager,
    connect: async () => ({
      discover: async () => ({ characteristic: () => ({ subscribe: async () => ({ values: (async function* () { await notificationWaiting; })(), remove: async () => undefined }), write: async () => undefined }) }),
      disconnect: async () => undefined,
      lifecycleEvents: (async function* () { await lossReady; yield { current: "lost" }; })(),
    }),
  };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => {
        if (command.kind === "terminal") terminalCommands.push(command);
        if (command.kind === "start") return [{ kind: "subscribe", actionId: 1, channel: "authentication" }];
        return [];
      },
      stop: async () => undefined,
    }),
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  const importing = controller.importSensor({ sensorName: "sensor", userActivation: true });
  await new Promise<void>(resolve => setImmediate(resolve));
  releaseLoss?.();
  await assert.rejects(importing, /link-lost/);
  assert.deepEqual(terminalCommands, [{ kind: "terminal", reason: "link-lost" }]);
});

test("selects the chooser once and retries the same peer", async () => {
  const fake = fakeManager();
  let chooses = 0;
  let connects = 0;
  let starts = 0;
  const pairingCodes: (string | null)[] = [];
  const manager = {
    ...fake.manager,
    choose: async () => { chooses += 1; return { id: "stable-peer", name: "G7" }; },
    connect: async () => {
      connects += 1;
      return {
        discover: async () => ({ characteristic: () => ({ subscribe: async () => ({ values: (async function* () {})(), remove: async () => undefined }), write: async () => undefined }) }),
        disconnect: async () => undefined,
      };
    },
  };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({ push: async (command: EngineCommand) => {
      if (command.kind !== "start") return [];
      pairingCodes.push(command.pairingCode);
      return ++starts === 1 ? [{ kind: "failure", category: "history-incomplete" }] : [{ kind: "complete", completeness: "complete" }];
    }, stop: async () => undefined }),
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 2, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await controller.importSensor({ sensorName: "sensor", userActivation: true, pairingCode: "1234" });

  assert.equal(chooses, 1);
  assert.equal(connects, 2);
  assert.deepEqual(pairingCodes, ["1234", "1234"]);
});

test("automatically selects the only authorized peer with a remembered credential without activation", async () => {
  const starts: EngineCommand[] = [];
  const peer = { id: "authorized-peer", name: "Remembered G7" };
  const fake = fakeManager();
  const manager = {
    ...fake.manager,
    choose: async () => { throw new Error("chooser must not run"); },
    peers: { authorized: async () => [peer] },
    connect: async (selected: { id: string }) => {
      assert.equal(selected.id, peer.id);
      return { discover: async () => ({ characteristic: () => ({ subscribe: async () => ({ values: (async function* () {})(), remove: async () => undefined }), write: async () => undefined }) }), disconnect: async () => undefined };
    },
  };
  const credential = new Uint8Array([6]);
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({ push: async (command: EngineCommand) => { starts.push(command); return command.kind === "start" ? [{ kind: "complete", completeness: "complete" }] : []; }, stop: async () => undefined }),
    loadCredential: async (peerId) => { assert.equal(peerId, peer.id); return credential; },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await controller.importSensor({ sensorName: "fallback name", userActivation: false, selection: "authorized" });

  const start = starts.find((command): command is Extract<EngineCommand, { kind: "start" }> => command.kind === "start");
  assert.equal(start?.sensorName, peer.name);
  assert.deepEqual(start?.credential, credential);
  assert.equal(start?.pairingCode, null);
});

test("does not guess when zero or multiple authorized peers have credentials", async () => {
  const fake = fakeManager();
  let authorized: readonly { id: string; name: string }[] = [];
  const manager = { ...fake.manager, choose: async () => { throw new Error("chooser must not run"); }, peers: { authorized: async () => authorized } };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({ push: async () => [], stop: async () => undefined }),
    loadCredential: async () => new Uint8Array([1]),
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await assert.rejects(controller.importSensor({ sensorName: "sensor", userActivation: false, selection: "authorized" }), /choose sensor/);
  authorized = [{ id: "peer-a", name: "A" }, { id: "peer-b", name: "B" }];
  const loadCredential = async (peerId: string) => peerId === "peer-a" ? new Uint8Array([1]) : new Uint8Array([2]);
  const multiple = createSensorController({ ...deps, loadCredential }, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });
  await assert.rejects(multiple.importSensor({ sensorName: "sensor", userActivation: false, selection: "authorized" }), /choose sensor/);
});

test("waits between authorized connection attempts until a sleeping sensor wakes", async () => {
  let connects = 0;
  const delays: number[] = [];
  const peer = { id: "sleeping-peer", name: "G7" };
  const fake = fakeManager();
  const manager = {
    ...fake.manager,
    choose: async () => { throw new Error("chooser must not run"); },
    peers: { authorized: async () => [peer] },
    connect: async () => {
      connects += 1;
      if (connects < 2) throw new Error("sensor asleep");
      return { discover: async () => ({ characteristic: () => ({ subscribe: async () => ({ values: (async function* () {})(), remove: async () => undefined }), write: async () => undefined }) }), disconnect: async () => undefined };
    },
  };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({ push: async (command: EngineCommand) => command.kind === "start" ? [{ kind: "complete", completeness: "complete" }] : [], stop: async () => undefined }),
    loadCredential: async () => new Uint8Array([1]),
    clock: () => 0,
    timer: { sleep: async delay => { delays.push(delay); } },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, authorizedRetryAttempts: 2, authorizedRetryDelayMs: 7, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await controller.importSensor({ sensorName: "sensor", userActivation: false, selection: "authorized" });

  assert.equal(connects, 2);
  assert.deepEqual(delays, [7]);
});

test("stops an authorized reconnect wait without another check", async () => {
  let connects = 0;
  let releaseDelay: (() => void) | undefined;
  const delayGate = new Promise<void>(resolve => { releaseDelay = resolve; });
  const fake = fakeManager();
  const manager = { ...fake.manager, choose: async () => { throw new Error("chooser must not run"); }, peers: { authorized: async () => [{ id: "sleeping-peer", name: "G7" }] }, connect: async () => { connects += 1; throw new Error("sensor asleep"); } };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({ push: async () => [], stop: async () => undefined }),
    loadCredential: async () => new Uint8Array([1]),
    clock: () => 0,
    timer: { sleep: async () => delayGate },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, authorizedRetryAttempts: 3, authorizedRetryDelayMs: 7, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  const importing = controller.importSensor({ sensorName: "sensor", userActivation: false, selection: "authorized" });
  await new Promise<void>(resolve => setImmediate(resolve));
  await controller.stop();
  releaseDelay?.();
  await assert.rejects(importing, /stopped/);
  assert.equal(connects, 1);
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

test("loads remembered credentials by chooser peer id before start while explicit credentials win", async () => {
  const fake = fakeManager();
  const lookedUp: string[] = [];
  const starts: (Uint8Array | null)[] = [];
  const remembered = new Uint8Array([7]);
  const explicit = new Uint8Array([8]);
  let runs = 0;
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => {
        if (command.kind !== "start") return [];
        starts.push(command.credential);
        return [{ kind: "complete", completeness: "partial" }];
      },
      stop: async () => undefined,
    }),
    loadCredential: async (peerId) => { lookedUp.push(peerId); return remembered; },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await controller.importSensor({ sensorName: "editable name", userActivation: true });
  runs += 1;
  await controller.importSensor({ sensorName: "another name", userActivation: true, credential: explicit });

  assert.deepEqual(lookedUp, ["peer"]);
  assert.deepEqual(starts, [remembered, explicit]);
  assert.equal(runs, 1);
});

test("awaits credential persistence and scopes it to the selected peer", async () => {
  const fake = fakeManager();
  const saves: Array<{ peerId: string; credential: Uint8Array }> = [];
  let saveFinished = false;
  const credential = new Uint8Array([9]);
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => command.kind === "start"
        ? [{ kind: "persist-credential", credential }, { kind: "complete", completeness: "complete" }]
        : [],
      stop: async () => undefined,
    }),
    saveCredential: async (peerId, value) => {
      saves.push({ peerId, credential: value });
      await Promise.resolve();
      saveFinished = true;
    },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await controller.importSensor({ sensorName: "editable name", userActivation: true });

  assert.deepEqual(saves, [{ peerId: "peer", credential }]);
  assert.equal(saveFinished, true);
});

test("does not persist stale actions after a retry or stop", async () => {
  const fake = fakeManager();
  const saves: string[] = [];
  let releaseStart: (() => void) | undefined;
  const startGate = new Promise<void>(resolve => { releaseStart = resolve; });
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => command.kind === "start"
        ? (await startGate, [{ kind: "persist-credential", credential: new Uint8Array([1]) }])
        : [],
      stop: async () => undefined,
    }),
    saveCredential: async (peerId) => { saves.push(peerId); },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 2, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  const importing = controller.importSensor({ sensorName: "sensor", userActivation: true });
  await Promise.resolve();
  await controller.stop();
  releaseStart?.();
  await assert.rejects(importing, /stopped/);
  assert.deepEqual(saves, []);
});

test("does not persist a notification action from a prior retry generation", async () => {
  let connects = 0;
  let starts = 0;
  let releaseOldFrame: (() => void) | undefined;
  const oldFrameGate = new Promise<void>(resolve => { releaseOldFrame = resolve; });
  const saves: string[] = [];
  const manager = {
    choose: async () => {
      return { id: "peer-1" };
    },
    connect: async () => {
      connects += 1;
      if (connects === 2) releaseOldFrame?.();
      const connectionAttempt = connects;
      return {
        discover: async () => ({
          characteristic: () => ({
            subscribe: async () => ({
              values: (async function* () {
                if (connectionAttempt === 1) { await oldFrameGate; yield { value: new Uint8Array([1]) }; }
              })(),
              remove: async () => undefined,
            }),
            write: async () => undefined,
          }),
        }),
        disconnect: async () => undefined,
      };
    },
    destroy: async () => undefined,
  };
  const deps: SensorControllerDependencies = {
    createManager: async () => manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => {
        if (command.kind === "start") { starts += 1; return [{ kind: "subscribe", actionId: 1, channel: "authentication" }]; }
        if (command.kind === "action-result" && starts === 1) return [{ kind: "failure", category: "history-incomplete" }];
        if (command.kind === "frame") return [{ kind: "persist-credential", credential: new Uint8Array([2]) }];
        return [{ kind: "complete", completeness: "complete" }];
      },
      stop: async () => undefined,
    }),
    saveCredential: async (peerId) => { saves.push(peerId); },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 2, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await controller.importSensor({ sensorName: "sensor", userActivation: true });
  await Promise.resolve();
  assert.deepEqual(saves, []);
});

test("fails closed when remembered credential lookup fails", async () => {
  const fake = fakeManager();
  let started = false;
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => {
        if (command.kind === "start") { started = true; return []; }
        return [];
      },
      stop: async () => undefined,
    }),
    loadCredential: async () => { throw new Error("vault unavailable"); },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await assert.rejects(controller.importSensor({ sensorName: "sensor", userActivation: true }), /vault unavailable/);
  assert.equal(started, false);
});

test("fails closed when credential persistence fails", async () => {
  const fake = fakeManager();
  const deps: SensorControllerDependencies = {
    createManager: async () => fake.manager,
    createEngine: async () => ({
      push: async (command: EngineCommand) => command.kind === "start"
        ? [{ kind: "reading", reading }, { kind: "persist-credential", credential: new Uint8Array([3]) }]
        : [],
      stop: async () => undefined,
    }),
    saveCredential: async () => { throw new Error("vault write failed"); },
    clock: () => 0,
    timer: { sleep: async () => undefined },
    entropy: () => new Uint8Array(),
  };
  const controller = createSensorController(deps, { maxAttempts: 1, serviceUuid: "service", channels: { authentication: "auth", control: "control", backfill: "backfill", "extra-data": "extra" } });

  await assert.rejects(controller.importSensor({ sensorName: "sensor", userActivation: true }), /vault write failed/);
});
