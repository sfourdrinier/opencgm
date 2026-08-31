// web/lib/sensor/reading-archive.ts

import type { ImportedSensorReading } from "./contracts";

const ARCHIVE_VERSION = 1;
const DEFAULT_DATABASE_NAME = "opencgm-reading-archive";
const STORE_NAME = "readings";

type ArchivePersistence = "persistent" | "session-only" | "memory";

export type ReadingArchiveSummary = {
  readonly sensorId: string;
  readonly count: number;
  readonly oldestAtMs: number;
  readonly newestAtMs: number;
  readonly latest: ImportedSensorReading;
};

export type ReadingArchiveOptions = {
  readonly databaseName?: string;
  readonly indexedDB?: IDBFactory;
  readonly persistence?: ArchivePersistence;
};

export interface ReadingArchive {
  readonly persistence: ArchivePersistence;
  save(reading: ImportedSensorReading): Promise<void>;
  ingest(readings: readonly ImportedSensorReading[]): Promise<void>;
  list(sensorId?: string): Promise<ImportedSensorReading[]>;
  summaries(): Promise<ReadingArchiveSummary[]>;
  forget(sensorId: string): Promise<void>;
  clear(): Promise<void>;
}

type StoredReading = {
  readonly schemaVersion: typeof ARCHIVE_VERSION;
  readonly key: string;
  readonly reading: ImportedSensorReading;
};

const sessionStores = new Map<string, Map<string, unknown>>();

/** Open the browser archive, falling back to a session-local store when needed. */
export async function createReadingArchive(options: ReadingArchiveOptions = {}): Promise<ReadingArchive> {
  const name = options.databaseName ?? DEFAULT_DATABASE_NAME;
  const requestedPersistence = options.persistence ?? "persistent";
  if (requestedPersistence !== "memory") {
    const factory = options.indexedDB === undefined
      ? (typeof globalThis.indexedDB === "undefined" ? undefined : globalThis.indexedDB)
      : options.indexedDB;
    if (factory) {
      try {
        return new IndexedDbReadingArchive(await IndexedDbReadingArchive.open(factory, name));
      } catch {
        // Browser privacy modes and blocked storage are expected. Keep imports usable.
      }
    }
  }
  const store = requestedPersistence === "memory"
    ? new Map<string, unknown>()
    : sessionStores.get(name) ?? new Map<string, unknown>();
  if (requestedPersistence !== "memory") sessionStores.set(name, store);
  return new MapReadingArchive(store, requestedPersistence === "memory" ? "memory" : "session-only");
}

class MapReadingArchive implements ReadingArchive {
  readonly persistence: ArchivePersistence;
  readonly #store: Map<string, unknown>;
  #mutation: Promise<void> = Promise.resolve();

  constructor(store: Map<string, unknown>, persistence: ArchivePersistence) {
    this.#store = store;
    this.persistence = persistence;
  }

  save(reading: ImportedSensorReading): Promise<void> {
    return this.#enqueue(() => {
      assertReading(reading);
      const key = readingKey(reading);
      const previous = validStored(this.#store.get(key));
      if (!previous || preferred(reading, previous.reading)) {
        this.#store.set(key, stored(reading));
      }
    });
  }

  async ingest(readings: readonly ImportedSensorReading[]): Promise<void> {
    for (const reading of readings) await this.save(reading);
  }

  async list(sensorId?: string): Promise<ImportedSensorReading[]> {
    const rows: ImportedSensorReading[] = [];
    for (const raw of this.#store.values()) {
      const row = validStored(raw);
      if (row && (sensorId === undefined || row.reading.sensorId === sensorId)) rows.push(copyReading(row.reading));
    }
    return sortReadings(rows);
  }

  async summaries(): Promise<ReadingArchiveSummary[]> {
    const grouped = new Map<string, ImportedSensorReading[]>();
    for (const row of await this.list()) {
      const readings = grouped.get(row.sensorId) ?? [];
      readings.push(row);
      grouped.set(row.sensorId, readings);
    }
    return [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([sensorId, readings]) => ({
      sensorId,
      count: readings.length,
      oldestAtMs: readings[0]!.atMs,
      newestAtMs: readings[readings.length - 1]!.atMs,
      latest: copyReading(readings[readings.length - 1]!),
    }));
  }

  forget(sensorId: string): Promise<void> {
    return this.#enqueue(() => {
      for (const [key, raw] of this.#store) {
        const row = validStored(raw);
        if ((row?.reading.sensorId ?? extractSensorId(raw)) === sensorId) this.#store.delete(key);
      }
    });
  }

  clear(): Promise<void> {
    return this.#enqueue(() => this.#store.clear());
  }

  #enqueue(action: () => void): Promise<void> {
    const next = this.#mutation.then(action);
    this.#mutation = next.catch(() => undefined);
    return next;
  }
}

class IndexedDbReadingArchive implements ReadingArchive {
  readonly persistence = "persistent" as const;
  readonly #database: IDBDatabase;
  #mutation: Promise<void> = Promise.resolve();

  constructor(database: IDBDatabase) {
    this.#database = database;
  }

  static async open(factory: IDBFactory, name: string): Promise<IDBDatabase> {
    return new Promise<IDBDatabase>((resolve, reject) => {
      const request = factory.open(name, ARCHIVE_VERSION);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(STORE_NAME)) {
          request.result.createObjectStore(STORE_NAME, { keyPath: "key" });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
      request.onblocked = () => reject(new Error("IndexedDB open blocked"));
    });
  }

  save(reading: ImportedSensorReading): Promise<void> {
    return this.#enqueue(async () => {
      assertReading(reading);
      const key = readingKey(reading);
      const existing = await this.request<unknown>("readonly", (store) => store.get(key));
      const previous = validStored(existing);
      if (!previous || preferred(reading, previous.reading)) {
        await this.request<IDBValidKey>("readwrite", (store) => store.put(stored(reading)), true);
      }
    });
  }

  async ingest(readings: readonly ImportedSensorReading[]): Promise<void> {
    for (const reading of readings) await this.save(reading);
  }

  async list(sensorId?: string): Promise<ImportedSensorReading[]> {
    const rawRows = await this.request<unknown[]>("readonly", (store) => store.getAll());
    return sortReadings(rawRows.flatMap((raw) => {
      const row = validStored(raw);
      return row && (sensorId === undefined || row.reading.sensorId === sensorId) ? [copyReading(row.reading)] : [];
    }));
  }

  async summaries(): Promise<ReadingArchiveSummary[]> {
    const grouped = new Map<string, ImportedSensorReading[]>();
    for (const row of await this.list()) {
      const readings = grouped.get(row.sensorId) ?? [];
      readings.push(row);
      grouped.set(row.sensorId, readings);
    }
    return [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([sensorId, readings]) => ({
      sensorId,
      count: readings.length,
      oldestAtMs: readings[0]!.atMs,
      newestAtMs: readings[readings.length - 1]!.atMs,
      latest: copyReading(readings[readings.length - 1]!),
    }));
  }

  forget(sensorId: string): Promise<void> {
    return this.#enqueue(async () => {
      const rawRows = await this.request<unknown[]>("readonly", (store) => store.getAll());
      const keys = rawRows.flatMap((raw) => {
        const row = validStored(raw);
        const matches = (row?.reading.sensorId ?? extractSensorId(raw)) === sensorId;
        const key = row?.key ?? (raw && typeof raw === "object" && typeof (raw as { key?: unknown }).key === "string" ? (raw as { key: string }).key : null);
        return matches && key ? [key] : [];
      });
      if (keys.length === 0) return;
      await this.request<undefined>("readwrite", (store) => {
        let last = store.delete(keys[0]!);
        for (const key of keys.slice(1)) last = store.delete(key);
        return last;
      }, true);
    });
  }

  clear(): Promise<void> {
    return this.#enqueue(() => this.request<undefined>("readwrite", (store) => store.clear(), true).then(() => undefined));
  }

  #enqueue(action: () => Promise<void>): Promise<void> {
    const next = this.#mutation.then(action);
    this.#mutation = next.catch(() => undefined);
    return next;
  }

  private request<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>, awaitCommit = false): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let transaction: IDBTransaction;
      let result: T;
      let settled = false;
      const resolveOnce = (value: T): void => {
        if (!settled) { settled = true; resolve(value); }
      };
      try {
        transaction = this.#database.transaction(STORE_NAME, mode);
        const request = action(transaction.objectStore(STORE_NAME));
        request.onsuccess = () => {
          result = request.result;
          if (!awaitCommit) resolveOnce(result);
        };
        request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
        transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed"));
        transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
        transaction.oncomplete = () => resolveOnce(result);
      } catch (error) {
        reject(error);
      }
    });
  }
}

function stored(reading: ImportedSensorReading): StoredReading {
  return { schemaVersion: ARCHIVE_VERSION, key: readingKey(reading), reading: copyReading(reading) };
}

function validStored(value: unknown): StoredReading | null {
  if (!value || typeof value !== "object") return null;
  const row = value as Partial<StoredReading>;
  if (row.schemaVersion !== ARCHIVE_VERSION || typeof row.key !== "string" || !row.reading || typeof row.reading !== "object") return null;
  const reading = row.reading as ImportedSensorReading;
  if (!validReading(reading) || row.key !== readingKey(reading)) return null;
  return { schemaVersion: ARCHIVE_VERSION, key: row.key, reading: copyReading(reading) };
}

function validReading(reading: ImportedSensorReading): boolean {
  return typeof reading.sensorId === "string" && reading.sensorId.length > 0
    && Number.isSafeInteger(reading.sensorSeconds) && reading.sensorSeconds >= 0
    && Number.isFinite(reading.atMs) && Number.isFinite(reading.receivedAtMs)
    && (reading.mgdl === null || typeof reading.mgdl === "number")
    && typeof reading.reliable === "boolean" && Number.isFinite(reading.algorithmState)
    && (reading.source === "live" || reading.source === "backfill");
}

function assertReading(reading: ImportedSensorReading): void {
  if (!validReading(reading)) throw new TypeError("Invalid imported sensor reading");
}

function readingKey(reading: Pick<ImportedSensorReading, "sensorId" | "sensorSeconds">): string {
  return JSON.stringify([reading.sensorId, reading.sensorSeconds]);
}

function preferred(candidate: ImportedSensorReading, current: ImportedSensorReading): boolean {
  const candidateUsable = candidate.reliable && candidate.mgdl !== null && Number.isFinite(candidate.mgdl);
  const currentUsable = current.reliable && current.mgdl !== null && Number.isFinite(current.mgdl);
  if (candidateUsable !== currentUsable) return candidateUsable;
  if (candidate.source !== current.source) return candidate.source === "live";
  if (candidate.receivedAtMs !== current.receivedAtMs) return candidate.receivedAtMs > current.receivedAtMs;
  return candidate.atMs > current.atMs;
}

function copyReading(reading: ImportedSensorReading): ImportedSensorReading {
  return {
    sensorId: reading.sensorId,
    sensorSeconds: reading.sensorSeconds,
    atMs: reading.atMs,
    receivedAtMs: reading.receivedAtMs,
    mgdl: reading.mgdl,
    reliable: reading.reliable,
    algorithmState: reading.algorithmState,
    source: reading.source,
  };
}

function sortReadings(readings: ImportedSensorReading[]): ImportedSensorReading[] {
  return readings.sort((a, b) => a.atMs - b.atMs || a.sensorId.localeCompare(b.sensorId) || a.sensorSeconds - b.sensorSeconds || a.receivedAtMs - b.receivedAtMs || sourceOrder(a.source) - sourceOrder(b.source));
}

function sourceOrder(source: ImportedSensorReading["source"]): number {
  return source === "backfill" ? 0 : 1;
}

function extractSensorId(value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const reading = (value as { reading?: unknown }).reading;
  return reading && typeof reading === "object" && typeof (reading as { sensorId?: unknown }).sensorId === "string"
    ? (reading as { sensorId: string }).sensorId
    : null;
}
