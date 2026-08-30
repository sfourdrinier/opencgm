import { createWebBleManager } from "unified-ble-manager/web";
import { loadSensorEngine } from "./engine";
import { normalizeSensorImport } from "./normalize";
import type {
  EngineAction,
  EngineCommand,
  ImportedSensorReading,
  SensorChannel,
  SensorImportMetadata,
  SensorImportResult,
} from "./contracts";

export type WebSensorSupport =
  | { readonly state: "supported" }
  | { readonly state: "unsupported"; readonly reason: "secure-context-required" | "web-bluetooth-unavailable" };

export function getWebSensorSupport(input: { readonly secureContext: boolean; readonly bluetooth: boolean; readonly origin?: string }): WebSensorSupport {
  const localhostException = input.origin !== undefined && /^http:\/\/localhost(?::\d+)?$/u.test(input.origin);
  if (!input.secureContext && !localhostException) return { state: "unsupported", reason: "secure-context-required" };
  if (!input.bluetooth) return { state: "unsupported", reason: "web-bluetooth-unavailable" };
  return { state: "supported" };
}

export type SensorPeer = { readonly id: string };
export type SensorSubscription = {
  readonly values: AsyncIterable<{ readonly value: Uint8Array }>;
  readonly remove: () => Promise<unknown>;
};
export type SensorCharacteristic = {
  readonly subscribe: () => Promise<SensorSubscription>;
  readonly write: (value: Uint8Array, options?: { readonly response?: "required" | "not-required" | "automatic" }) => Promise<unknown>;
};
export type SensorDatabase = {
  readonly characteristic: (serviceUuid: string, characteristicUuid: string) => SensorCharacteristic;
};
export type SensorConnectionEvent = { readonly current?: string; readonly cause?: string };
export type SensorConnection = {
  readonly discover: () => Promise<SensorDatabase>;
  readonly disconnect: () => Promise<unknown>;
  readonly lifecycleEvents?: AsyncIterable<SensorConnectionEvent>;
};
export type SensorManager = {
  readonly choose: (request: SensorChooserRequest) => Promise<SensorPeer>;
  readonly connect: (peer: SensorPeer) => Promise<SensorConnection>;
  readonly destroy: () => Promise<unknown>;
};
export type SensorChooserRequest = {
  readonly filters: readonly { readonly serviceUuids: readonly string[] }[];
  readonly optionalServices: readonly string[];
};
export type SensorEngineRuntime = {
  readonly push: (command: EngineCommand) => Promise<readonly EngineAction[]>;
  readonly stop: () => Promise<void>;
};
export type SensorTimer = { readonly sleep: (delayMs: number) => Promise<void> };

export type SensorControllerDependencies = {
  readonly createManager: () => Promise<SensorManager>;
  readonly createEngine: () => Promise<SensorEngineRuntime>;
  readonly loadCredential?: (peerId: string) => Promise<Uint8Array | null>;
  readonly saveCredential?: (peerId: string, credential: Uint8Array) => Promise<void>;
  readonly onPeerSelected?: (peerId: string) => void;
  readonly clock: () => number;
  readonly timer: SensorTimer;
  readonly entropy: (byteCount: number) => Uint8Array;
};

export type SensorCredentialCallbacks = Pick<SensorControllerDependencies, "loadCredential" | "saveCredential" | "onPeerSelected">;

export type SensorControllerOptions = {
  readonly serviceUuid: string;
  readonly channels: Readonly<Record<SensorChannel, string>>;
  readonly optionalServices?: readonly string[];
  readonly maxAttempts?: number;
};
export type SensorImportRequest = {
  readonly sensorName: string;
  readonly sensorId?: string;
  readonly userActivation: boolean;
  readonly credential?: Uint8Array | null;
  readonly pairingCode?: string | null;
  readonly certificateBundle?: Uint8Array | null;
};
export type SensorController = {
  readonly importSensor: (request: SensorImportRequest) => Promise<SensorImportResult>;
  readonly stop: () => Promise<void>;
};

export function createDefaultSensorController(
  options: SensorControllerOptions,
  credentialCallbacks: SensorCredentialCallbacks = {},
): SensorController {
  return createSensorController(
    {
      createManager: async () => {
        const ble = await createWebBleManager();
        let selected: Awaited<ReturnType<typeof ble.choose>> | null = null;
        return {
          choose: async (request: SensorChooserRequest) => {
            const peer = await ble.choose({ filters: request.filters, optionalServices: request.optionalServices });
            selected = peer;
            return { id: peer.id };
          },
          connect: async () => {
            if (selected === null) throw new Error("no chooser selection");
            const connection = await ble.connect(selected);
            return {
              discover: async () => {
                const database = await connection.discover();
                return {
                  characteristic: (serviceUuid: string, characteristicUuid: string) => {
                    const characteristic = database.characteristic(serviceUuid, characteristicUuid);
                    return {
                      subscribe: async () => {
                        const subscription = await characteristic.subscribe();
                        return {
                          values: (async function* () {
                            for await (const event of subscription.values) {
                              if ("value" in event && event.value instanceof Uint8Array) yield { value: event.value };
                            }
                          })(),
                          remove: () => subscription.remove(),
                        };
                      },
                      write: (value: Uint8Array, writeOptions?: { readonly response?: "required" | "not-required" | "automatic" }) => characteristic.write(value, writeOptions),
                    };
                  },
                };
              },
              disconnect: () => connection.disconnect(),
              lifecycleEvents: connection.lifecycleEvents,
            };
          },
          destroy: () => ble.destroy(),
        };
      },
      createEngine: async () => {
        const loaded = await loadSensorEngine();
        return { push: loaded.push, stop: loaded.stop };
      },
      ...credentialCallbacks,
      clock: () => Date.now(),
      timer: { sleep: async delayMs => new Promise(resolve => setTimeout(resolve, delayMs)) },
      entropy: byteCount => crypto.getRandomValues(new Uint8Array(byteCount)),
    },
    options,
  );
}

export function createSensorController(
  dependencies: SensorControllerDependencies,
  options: SensorControllerOptions,
): SensorController {
  let manager: SensorManager | null = null;
  let engine: SensorEngineRuntime | null = null;
  let stopped = false;
  let runPromise: Promise<SensorImportResult> | null = null;
  const disconnectPromises = new WeakMap<SensorConnection, Promise<unknown>>();
  let destroyPromise: Promise<unknown> | null = null;

  const stop = async (): Promise<void> => {
    stopped = true;
    await cleanupSubscriptions();
    if (engine !== null) await stopEngineOnce();
    if (manager !== null) await destroyManagerOnce();
  };

  const importSensor = async (request: SensorImportRequest): Promise<SensorImportResult> => {
    if (runPromise !== null) return runPromise;
    if (!request.userActivation) throw new Error("sensor chooser requires transient user activation");
    if (stopped) throw new Error("sensor controller is stopped");
    runPromise = run(request).finally(() => { runPromise = null; });
    return runPromise;
  };

  async function run(request: SensorImportRequest): Promise<SensorImportResult> {
    manager = await dependencies.createManager();
    engine = await dependencies.createEngine();
    if (stopped) throw new Error("sensor controller stopped");
    destroyPromise = null;
    recordsForCurrentRun = [];
    metadataForCurrentRun = null;
    completionForCurrentRun = false;
    let completeness: SensorImportResult["completeness"] = "partial";
    const warnings: string[] = [];
    const maxAttempts = Math.max(1, options.maxAttempts ?? 2);
    credentialPersistenceFailedForCurrentRun = false;
    try {
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        let connection: SensorConnection | null = null;
        const generation = { value: attempt };
        try {
          const peer = await manager!.choose({
            filters: [{ serviceUuids: [options.serviceUuid] }],
            optionalServices: [...(options.optionalServices ?? [])],
          });
          dependencies.onPeerSelected?.(peer.id);
          if (stopped) throw new Error("sensor controller stopped");
          const rememberedCredential = request.credential === undefined || request.credential === null
            ? await dependencies.loadCredential?.(peer.id) ?? null
            : null;
          if (stopped) throw new Error("sensor controller stopped");
          connection = await manager!.connect(peer);
          if (stopped) throw new Error("sensor controller stopped");
          const database = await connection.discover();
          if (stopped) throw new Error("sensor controller stopped");
          if (connection.lifecycleEvents !== undefined) pumpLifecycle(connection.lifecycleEvents, () => { generation.value += 1; });
          const startActions = await engine.push({
            kind: "start",
            nowMs: dependencies.clock(),
            sensorName: request.sensorName,
            credential: request.credential ?? rememberedCredential,
            pairingCode: request.pairingCode ?? null,
            certificateBundle: request.certificateBundle ?? null,
          });
          await processActions(startActions, database, generation, generation.value, peer.id);
          completeness = "complete";
          break;
        } catch (error) {
          warnings.push(error instanceof Error ? error.message : "sensor operation failed");
          if (connection !== null) await disconnectOnce(connection);
          if (credentialPersistenceFailedForCurrentRun || (attempt === maxAttempts && recordsForCurrentRun.length === 0)) throw error;
        } finally {
          generation.value += 1;
          if (connection !== null) await disconnectOnce(connection);
        }
      }
      if (!completionForCurrentRun) completeness = "partial";
      const metadata = metadataFor(request, recordsForCurrentRun, metadataForCurrentRun);
      return normalizeSensorImport(recordsForCurrentRun, metadata, completeness, warnings);
    } finally {
      await cleanupSubscriptions();
      await stopEngineOnce();
      await destroyManagerOnce();
    }
  }

  async function processActions(actions: readonly EngineAction[], database: SensorDatabase, generation: { value: number }, expectedGeneration = generation.value, peerId = ""): Promise<void> {
    for (const action of actions) {
      if (stopped) throw new Error("sensor controller stopped");
      if (generation.value !== expectedGeneration) throw new Error("stale sensor action");
      if (action.kind === "reading") recordsForCurrentRun.push(action.reading);
      else if (action.kind === "metadata") metadataForCurrentRun = action.metadata;
      else if (action.kind === "complete") completionForCurrentRun = action.completeness === "complete";
      else if (action.kind === "failure") throw new Error(action.category);
      else if (action.kind === "request-os-pair") throw new Error("browser Web Bluetooth does not support OS pairing");
      else if (action.kind === "need-entropy") {
        const next = await engine!.push({ kind: "action-result", actionId: action.actionId, ok: true, bytes: dependencies.entropy(action.byteCount) });
        await processActions(next, database, generation, expectedGeneration, peerId);
      } else if (action.kind === "subscribe") {
        const characteristic = database.characteristic(options.serviceUuid, options.channels[action.channel]);
        const subscription = await characteristic.subscribe();
        subscriptions.push(subscription);
        pumpNotifications(subscription.values, database, generation, action.channel, peerId, expectedGeneration);
        const next = await engine!.push({ kind: "action-result", actionId: action.actionId, ok: true, bytes: new Uint8Array() });
        await processActions(next, database, generation, expectedGeneration, peerId);
      } else if (action.kind === "write") {
        const characteristic = database.characteristic(options.serviceUuid, options.channels[action.channel]);
        await characteristic.write(action.bytes, { response: action.response });
        if (action.delayAfterMs > 0) await dependencies.timer.sleep(action.delayAfterMs);
        const next = await engine!.push({ kind: "action-result", actionId: action.actionId, ok: true, bytes: new Uint8Array() });
        await processActions(next, database, generation, expectedGeneration, peerId);
      } else if (action.kind === "persist-credential") {
        if (dependencies.saveCredential !== undefined) {
          try {
            await dependencies.saveCredential(peerId, new Uint8Array(action.credential));
          } catch (error) {
            credentialPersistenceFailedForCurrentRun = true;
            throw error;
          }
        }
      }
    }
  }

  // Per-run mutable state is kept in the closure to preserve all records on a retry.
  let recordsForCurrentRun: ImportedSensorReading[] = [];
  let metadataForCurrentRun: SensorImportMetadata | null = null;
  let completionForCurrentRun = false;
  let credentialPersistenceFailedForCurrentRun = false;
  const subscriptions: SensorSubscription[] = [];

  async function pumpNotifications(values: AsyncIterable<{ readonly value: Uint8Array }>, database: SensorDatabase, generation: { value: number }, channel: SensorChannel, peerId: string, expectedGeneration: number): Promise<void> {
    try {
      for await (const event of values) {
        if (generation.value !== expectedGeneration || stopped || engine === null) throw new Error("stale sensor notification");
        const currentEngine = engine;
        const actions = await currentEngine.push({ kind: "frame", channel, bytes: new Uint8Array(event.value), nowMs: dependencies.clock() });
        await processActions(actions, database, generation, expectedGeneration, peerId);
      }
    } catch {
      // Notification streams are best-effort after the direct operation's result;
      // the retained records remain available to the partial import result.
    }
  }

  async function pumpLifecycle(events: AsyncIterable<SensorConnectionEvent>, invalidate: () => void): Promise<void> {
    try {
      for await (const event of events) {
        if (event.current === "disconnected" || event.cause === "link-lost") invalidate();
      }
    } catch {
      invalidate();
    }
  }

  async function disconnectOnce(connection: SensorConnection): Promise<void> {
    let promise = disconnectPromises.get(connection);
    if (promise === undefined) {
      promise = connection.disconnect();
      disconnectPromises.set(connection, promise);
    }
    await promise;
  }

  async function destroyManagerOnce(): Promise<void> {
    if (manager !== null && destroyPromise === null) destroyPromise = manager.destroy();
    if (destroyPromise !== null) await destroyPromise;
  }

  async function stopEngineOnce(): Promise<void> {
    if (engine !== null) {
      const current = engine;
      engine = null;
      await current.stop();
    }
  }

  async function cleanupSubscriptions(): Promise<void> {
    for (const subscription of subscriptions.splice(0)) await subscription.remove();
  }

  function metadataFor(request: SensorImportRequest, records: readonly ImportedSensorReading[], existing?: SensorImportMetadata | null): SensorImportMetadata {
    const sensorId = existing?.sensorId ?? request.sensorId ?? records[0]?.sensorId ?? request.sensorName;
    const timestamps = records.map(record => record.atMs).filter(Number.isFinite);
    return existing === null || existing === undefined
      ? { sensorId, activatedAtMs: null, firmware: null, oldestAtMs: timestamps.length ? Math.min(...timestamps) : null, newestAtMs: timestamps.length ? Math.max(...timestamps) : null, readingCount: records.length, duplicateCount: 0 }
      : { ...existing, readingCount: records.length, oldestAtMs: timestamps.length ? Math.min(...timestamps) : existing.oldestAtMs, newestAtMs: timestamps.length ? Math.max(...timestamps) : existing.newestAtMs };
  }

  return { importSensor, stop };
}
