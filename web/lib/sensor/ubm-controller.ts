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

export class SensorSelectionError extends Error {
  readonly code = "choose-sensor" as const;

  constructor() {
    super("choose sensor");
    this.name = "SensorSelectionError";
  }
}

const DEFAULT_AUTHORIZED_RETRY_DELAY_MS = 15_000;
const DEFAULT_AUTHORIZED_RETRY_ATTEMPTS = 25;
const DEFAULT_CONNECT_TIMEOUT_MS = 20_000;

function isUnsettledWebConnectTimeout(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  return Reflect.get(error, "code") === "operation.timed-out"
    && Reflect.get(error, "operation") === "web-connection.connect";
}

export function getWebSensorSupport(input: { readonly secureContext: boolean; readonly bluetooth: boolean; readonly origin?: string }): WebSensorSupport {
  const localhostException = input.origin !== undefined && /^http:\/\/localhost(?::\d+)?$/u.test(input.origin);
  if (!input.secureContext && !localhostException) return { state: "unsupported", reason: "secure-context-required" };
  if (!input.bluetooth) return { state: "unsupported", reason: "web-bluetooth-unavailable" };
  return { state: "supported" };
}

export type SensorPeer = { readonly id: string; readonly name?: string | null };
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
  readonly peers?: { readonly authorized: () => Promise<readonly SensorPeer[]> };
  readonly connect: (peer: SensorPeer, options?: { readonly timeoutMs: number }) => Promise<SensorConnection>;
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

type AttemptTerminal = {
  readonly promise: Promise<void>;
  readonly resolve: () => void;
  readonly reject: (reason: unknown) => void;
  settled: boolean;
  waiting: boolean;
  streamEnded: boolean;
  terminalSent: boolean;
};

export type SensorControllerDependencies = {
  readonly createManager: () => Promise<SensorManager>;
  readonly createEngine: () => Promise<SensorEngineRuntime>;
  readonly loadCredential?: (peerId: string) => Promise<Uint8Array | null>;
  readonly saveCredential?: (peerId: string, credential: Uint8Array) => Promise<void>;
  readonly onPeerSelected?: (peerId: string) => void;
  readonly onReading?: (reading: ImportedSensorReading) => Promise<void> | void;
  readonly clock: () => number;
  readonly timer: SensorTimer;
  readonly entropy: (byteCount: number) => Uint8Array;
};

export type SensorCredentialCallbacks = Pick<SensorControllerDependencies, "loadCredential" | "saveCredential" | "onPeerSelected" | "onReading">;

export type SensorControllerOptions = {
  readonly serviceUuid: string;
  readonly chooserServiceUuid?: string;
  readonly channels: Readonly<Record<SensorChannel, string>>;
  readonly optionalServices?: readonly string[];
  readonly maxAttempts?: number;
  /** Delay between chooser reconnect attempts. Authorized reconnects use their dedicated delay. */
  readonly retryDelayMs?: number;
  readonly authorizedRetryDelayMs?: number;
  readonly authorizedRetryAttempts?: number;
  readonly connectTimeoutMs?: number;
};
export type SensorImportRequest = {
  readonly sensorName: string;
  readonly sensorId?: string;
  readonly userActivation: boolean;
  readonly credential?: Uint8Array | null;
  readonly pairingCode?: string | null;
  readonly certificateBundle?: Uint8Array | null;
  readonly selection?: "chooser" | "authorized";
  readonly historyStartSeconds?: number | null;
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
        return {
          choose: async (request: SensorChooserRequest) => {
            const peer = await ble.choose({ filters: request.filters, optionalServices: request.optionalServices });
            return { id: peer.id, name: peer.name };
          },
          peers: { authorized: async () => (await ble.peers.authorized()).map(peer => ({ id: peer.id, name: peer.name })) },
          connect: async (peer, connectOptions) => {
            const connection = await ble.connect(peer.id, { timeoutMs: connectOptions?.timeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS });
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
  let activeTerminal: AttemptTerminal | null = null;
  const disconnectPromises = new WeakMap<SensorConnection, Promise<unknown>>();
  let destroyPromise: Promise<unknown> | null = null;

  const stop = async (): Promise<void> => {
    stopped = true;
    activeTerminal?.reject(new Error("sensor controller stopped"));
    await cleanupSubscriptions();
    if (engine !== null) await stopEngineOnce();
    if (manager !== null) await destroyManagerOnce();
  };

  const importSensor = async (request: SensorImportRequest): Promise<SensorImportResult> => {
    if (runPromise !== null) return runPromise;
    if ((request.selection ?? "chooser") === "chooser" && !request.userActivation) throw new Error("sensor chooser requires transient user activation");
    if (stopped) throw new Error("sensor controller is stopped");
    runPromise = run(request).finally(() => { runPromise = null; });
    return runPromise;
  };

  async function run(request: SensorImportRequest): Promise<SensorImportResult> {
    manager = await dependencies.createManager();
    destroyPromise = null;
    recordsForCurrentRun = [];
    metadataForCurrentRun = null;
    completionForCurrentRun = false;
    let completeness: SensorImportResult["completeness"] = "partial";
    const warnings: string[] = [];
    const maxAttempts = request.selection === "authorized"
      ? Math.max(1, options.authorizedRetryAttempts ?? DEFAULT_AUTHORIZED_RETRY_ATTEMPTS)
      : Math.max(1, options.maxAttempts ?? 2);
    const authorizedRetryDelayMs = Math.max(0, options.authorizedRetryDelayMs ?? DEFAULT_AUTHORIZED_RETRY_DELAY_MS);
    const chooserRetryDelayMs = Math.max(0, options.retryDelayMs ?? 0);
    const connectTimeoutMs = options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS;
    credentialPersistenceFailedForCurrentRun = false;
    readingPersistenceFailedForCurrentRun = false;
    try {
      const selected = request.selection === "authorized"
        ? await selectAuthorizedPeer()
        : { peer: await manager!.choose({
            filters: [{ serviceUuids: [options.chooserServiceUuid ?? options.serviceUuid] }],
            optionalServices: [...new Set([...(options.optionalServices ?? []), options.serviceUuid])],
          }), credential: null };
      const peer = selected.peer;
      dependencies.onPeerSelected?.(peer.id);
      if (stopped) throw new Error("sensor controller stopped");
      const credential = selected.credential ?? (request.credential === undefined || request.credential === null
        ? await dependencies.loadCredential?.(peer.id) ?? null
        : null);
      if (stopped) throw new Error("sensor controller stopped");
      for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        let connection: SensorConnection | null = null;
        let attemptEngine: SensorEngineRuntime | null = null;
        const terminal = createAttemptTerminal();
        activeTerminal = terminal;
        const generation = { value: attempt };
        try {
          if (attempt > 1) {
            await dependencies.timer.sleep(request.selection === "authorized" ? authorizedRetryDelayMs : chooserRetryDelayMs);
            if (stopped) throw new Error("sensor controller stopped");
          }
          connection = await manager!.connect(peer, { timeoutMs: connectTimeoutMs });
          if (stopped) throw new Error("sensor controller stopped");
          const database = await connection.discover();
          if (stopped) throw new Error("sensor controller stopped");
          attemptEngine = await dependencies.createEngine();
          engine = attemptEngine;
          if (stopped) throw new Error("sensor controller stopped");
          if (connection.lifecycleEvents !== undefined) pumpLifecycle(connection.lifecycleEvents, database, attemptEngine, generation, terminal, peer.id);
          const startActions = await attemptEngine.push({
            kind: "start",
            nowMs: dependencies.clock(),
            sensorName: peer.name?.trim() || request.sensorName,
            credential: request.credential ?? credential,
            pairingCode: request.selection === "authorized" ? null : request.pairingCode ?? null,
            certificateBundle: request.certificateBundle ?? null,
            historyStartSeconds: request.historyStartSeconds ?? null,
          });
          await processActions(startActions, database, attemptEngine, generation, terminal, generation.value, peer.id);
          if (terminal.waiting && !terminal.settled) {
            if (terminal.streamEnded) terminal.reject(new Error("connection ended before completion"));
            else await terminal.promise;
          }
          completeness = "complete";
          break;
        } catch (error) {
          terminal.reject(error);
          warnings.push(error instanceof Error ? error.message : "sensor operation failed");
          if (connection !== null) await disconnectOnce(connection);
          if (connection === null && isUnsettledWebConnectTimeout(error)) throw error;
          if (credentialPersistenceFailedForCurrentRun || readingPersistenceFailedForCurrentRun || (attempt === maxAttempts && recordsForCurrentRun.length === 0)) throw error;
        } finally {
          generation.value += 1;
          if (connection !== null) await disconnectOnce(connection);
          await cleanupSubscriptions();
          if (engine === attemptEngine) await stopEngineOnce();
          activeTerminal = null;
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

  function createAttemptTerminal(): AttemptTerminal {
    let resolvePromise!: () => void;
    let rejectPromise!: (reason: unknown) => void;
    const promise = new Promise<void>((resolve, reject) => { resolvePromise = resolve; rejectPromise = reject; });
    // The direct action path can reject before a subscription makes the run await this promise.
    void promise.catch(() => undefined);
    const terminal: AttemptTerminal = {
      promise,
      resolve: () => { if (!terminal.settled) { terminal.settled = true; resolvePromise(); } },
      reject: reason => { if (!terminal.settled) { terminal.settled = true; rejectPromise(reason); } },
      settled: false,
      waiting: false,
      streamEnded: false,
      terminalSent: false,
    };
    return terminal;
  }

  async function selectAuthorizedPeer(): Promise<{ readonly peer: SensorPeer; readonly credential: Uint8Array }> {
    const peers = manager?.peers;
    if (peers === undefined) throw new SensorSelectionError();
    if (stopped) throw new Error("sensor controller stopped");
    let authorized: readonly SensorPeer[];
    try {
      authorized = await peers.authorized();
    } catch {
      throw new SensorSelectionError();
    }
    const eligible: Array<{ readonly peer: SensorPeer; readonly credential: Uint8Array }> = [];
    for (const peer of authorized) {
      try {
        const credential = await dependencies.loadCredential?.(peer.id);
        if (credential !== null && credential !== undefined) eligible.push({ peer, credential });
      } catch {
        // A vault failure cannot authorize a reconnect.
      }
    }
    if (eligible.length !== 1) throw new SensorSelectionError();
    return eligible[0]!;
  }

  async function processActions(actions: readonly EngineAction[], database: SensorDatabase, runtime: SensorEngineRuntime, generation: { value: number }, terminal: AttemptTerminal, expectedGeneration = generation.value, peerId = ""): Promise<void> {
    for (const action of actions) {
      if (stopped) { const error = new Error("sensor controller stopped"); terminal.reject(error); throw error; }
      if (generation.value !== expectedGeneration) { const error = new Error("stale sensor action"); terminal.reject(error); throw error; }
      if (action.kind === "reading") {
        recordsForCurrentRun.push(action.reading);
        try {
          await dependencies.onReading?.(action.reading);
        } catch (error) {
          readingPersistenceFailedForCurrentRun = true;
          throw error;
        }
      }
      else if (action.kind === "metadata") metadataForCurrentRun = action.metadata;
      else if (action.kind === "complete") { completionForCurrentRun = action.completeness === "complete"; terminal.resolve(); }
      else if (action.kind === "failure") { const error = new Error(action.category); terminal.reject(error); throw error; }
      else if (action.kind === "request-os-pair") { const error = new Error("browser Web Bluetooth does not support OS pairing"); terminal.reject(error); throw error; }
      else if (action.kind === "need-entropy") {
        const next = await runtime.push({ kind: "action-result", actionId: action.actionId, ok: true, bytes: dependencies.entropy(action.byteCount) });
        await processActions(next, database, runtime, generation, terminal, expectedGeneration, peerId);
      } else if (action.kind === "subscribe") {
        const characteristic = database.characteristic(options.serviceUuid, options.channels[action.channel]);
        const subscription = await characteristic.subscribe();
        subscriptions.push(subscription);
        terminal.waiting = true;
        pumpNotifications(subscription.values, database, runtime, generation, terminal, action.channel, peerId, expectedGeneration);
        const next = await runtime.push({ kind: "action-result", actionId: action.actionId, ok: true, bytes: new Uint8Array() });
        await processActions(next, database, runtime, generation, terminal, expectedGeneration, peerId);
      } else if (action.kind === "write") {
        const characteristic = database.characteristic(options.serviceUuid, options.channels[action.channel]);
        await characteristic.write(action.bytes, { response: action.response });
        if (action.delayAfterMs > 0) await dependencies.timer.sleep(action.delayAfterMs);
        const next = await runtime.push({ kind: "action-result", actionId: action.actionId, ok: true, bytes: new Uint8Array() });
        await processActions(next, database, runtime, generation, terminal, expectedGeneration, peerId);
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
  let readingPersistenceFailedForCurrentRun = false;
  const subscriptions: SensorSubscription[] = [];

  async function pumpNotifications(values: AsyncIterable<{ readonly value: Uint8Array }>, database: SensorDatabase, runtime: SensorEngineRuntime, generation: { value: number }, terminal: AttemptTerminal, channel: SensorChannel, peerId: string, expectedGeneration: number): Promise<void> {
    try {
      for await (const event of values) {
        if (generation.value !== expectedGeneration || stopped) throw new Error("stale sensor notification");
        const actions = await runtime.push({ kind: "frame", channel, bytes: new Uint8Array(event.value), nowMs: dependencies.clock() });
        await processActions(actions, database, runtime, generation, terminal, expectedGeneration, peerId);
      }
      terminal.streamEnded = true;
      if (terminal.waiting && !terminal.settled) terminal.reject(new Error("connection ended before completion"));
    } catch (error) {
      terminal.reject(error);
    }
  }

  async function pumpLifecycle(events: AsyncIterable<SensorConnectionEvent>, database: SensorDatabase, runtime: SensorEngineRuntime, generation: { value: number }, terminal: AttemptTerminal, peerId: string): Promise<void> {
    try {
      for await (const event of events) {
        if (event.current === "lost" || event.current === "disconnected" || event.cause === "peer-link-loss" || event.cause === "adapter-loss" || event.cause === "backend-failure") {
          generation.value += 1;
          if (!terminal.terminalSent) {
            terminal.terminalSent = true;
            try {
              const actions = await runtime.push({ kind: "terminal", reason: "link-lost" });
              await processActions(actions, database, runtime, generation, terminal, generation.value, peerId);
            } catch {
              // The link is already lost; the terminal waiter still must unblock.
            }
          }
          terminal.reject(new Error("link-lost"));
          return;
        }
      }
    } catch {
      generation.value += 1;
      terminal.reject(new Error("link-lost"));
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
      ? { sensorId, activatedAtMs: null, firmware: null, oldestAtMs: timestamps.length ? Math.min(...timestamps) : null, newestAtMs: timestamps.length ? Math.max(...timestamps) : null, readingCount: records.length, duplicateCount: 0, historyCompletedThroughSeconds: null }
      : { ...existing, readingCount: records.length, oldestAtMs: timestamps.length ? Math.min(...timestamps) : existing.oldestAtMs, newestAtMs: timestamps.length ? Math.max(...timestamps) : existing.newestAtMs };
  }

  return { importSensor, stop };
}
