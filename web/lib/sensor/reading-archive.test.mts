// web/lib/sensor/reading-archive.test.mts

import assert from "node:assert/strict";
import test from "node:test";
import type { ImportedSensorReading } from "./contracts";
import { createReadingArchive, type ReadingArchive } from "./reading-archive";

type FakeRequest<T> = IDBRequest<T>;

class TestRequest<T> {
  result!: T;
  error: DOMException | null = null;
  onsuccess: ((this: IDBRequest<T>, event: Event) => unknown) | null = null;
  onerror: ((this: IDBRequest<T>, event: Event) => unknown) | null = null;
  onupgradeneeded: ((this: IDBOpenDBRequest, event: IDBVersionChangeEvent) => unknown) | null = null;
  onblocked: ((this: IDBOpenDBRequest, event: Event) => unknown) | null = null;

  succeed(result: T): void {
    this.result = result;
    queueMicrotask(() => this.onsuccess?.call(this as unknown as IDBRequest<T>, new Event("success")));
  }
}

class TestStore {
  constructor(readonly rows = new Map<string, unknown>(), private readonly transaction: TestTransaction | null = null) {}
  get(key: IDBValidKey): TestRequest<unknown> {
    const request = new TestRequest<unknown>(); request.succeed(this.rows.get(String(key))); return request;
  }
  getAll(): TestRequest<unknown[]> {
    const request = new TestRequest<unknown[]>(); request.succeed([...this.rows.values()]); return request;
  }
  put(value: { key: string }): TestRequest<IDBValidKey> {
    this.rows.set(value.key, value); const request = new TestRequest<IDBValidKey>(); request.succeed(value.key); this.transaction?.completeSoon(); return request;
  }
  delete(key: IDBValidKey): TestRequest<undefined> {
    this.rows.delete(String(key)); const request = new TestRequest<undefined>(); request.succeed(undefined); this.transaction?.completeSoon(); return request;
  }
  clear(): TestRequest<undefined> {
    this.rows.clear(); const request = new TestRequest<undefined>(); request.succeed(undefined); this.transaction?.completeSoon(); return request;
  }
}

class TestTransaction {
  oncomplete: (() => unknown) | null = null;
  onerror: (() => unknown) | null = null;
  onabort: (() => unknown) | null = null;
  readonly store: TestStore;
  constructor(rows: Map<string, unknown>, private readonly abort = false) { this.store = new TestStore(rows, this); }
  objectStore(): TestStore { return this.store; }
  completeSoon(): void {
    queueMicrotask(() => {
      if (this.abort) this.onabort?.(); else this.oncomplete?.();
    });
  }
}

class TestDatabase {
  readonly store = new TestStore();
  readonly objectStoreNames = { contains: (name: string) => name === "readings" } as DOMStringList;
  abortNextWrite = false;
  createObjectStore(): TestStore { return this.store; }
  transaction(_name: string, mode: IDBTransactionMode): TestTransaction {
    const abort = mode === "readwrite" && this.abortNextWrite;
    if (abort) this.abortNextWrite = false;
    return new TestTransaction(this.store.rows, abort);
  }
  inject(value: unknown): void { this.store.rows.set(`corrupt-${this.store.rows.size}`, value); }
}

class TestIndexedDb {
  readonly databases = new Map<string, TestDatabase>();
  open(name: string): FakeRequest<IDBDatabase> {
    const request = new TestRequest<IDBDatabase>();
    const database = this.databases.get(name) ?? new TestDatabase();
    const fresh = !this.databases.has(name); this.databases.set(name, database);
    queueMicrotask(() => {
      request.result = database as unknown as IDBDatabase;
      if (fresh) request.onupgradeneeded?.call(request as unknown as IDBOpenDBRequest, new Event("upgradeneeded") as IDBVersionChangeEvent);
      request.succeed(database as unknown as IDBDatabase);
    });
    return request as unknown as FakeRequest<IDBDatabase>;
  }
  inject(name: string, value: unknown): void { this.databases.get(name)?.inject(value); }
  abortNextWrite(name: string): void { this.databases.get(name)!.abortNextWrite = true; }
}

const reading = (overrides: Partial<ImportedSensorReading> = {}): ImportedSensorReading => ({
  sensorId: "sensor-a",
  sensorSeconds: 10,
  atMs: 1_000,
  receivedAtMs: 1_100,
  mgdl: 110,
  reliable: true,
  algorithmState: 3,
  source: "backfill",
  ...overrides,
});

async function memoryArchive(name: string): Promise<ReadingArchive> {
  return createReadingArchive({ databaseName: `test-${name}`, persistence: "memory" });
}

test("persists readings between archive instances when IndexedDB is available", async () => {
  const indexedDB = new TestIndexedDb();
  const first = await createReadingArchive({ databaseName: "test-persistence", indexedDB: indexedDB as unknown as IDBFactory });
  await first.save(reading());
  const second = await createReadingArchive({ databaseName: "test-persistence", indexedDB: indexedDB as unknown as IDBFactory });
  assert.deepEqual(await second.list(), [reading()]);
});

test("does not report an IndexedDB write until its transaction commits", async () => {
  const indexedDB = new TestIndexedDb();
  const archive = await createReadingArchive({ databaseName: "test-commit", indexedDB: indexedDB as unknown as IDBFactory });
  indexedDB.abortNextWrite("test-commit");
  assert.equal(indexedDB.databases.get("test-commit")!.abortNextWrite, true);
  await assert.rejects(archive.save(reading()), /aborted/);
});

test("uses an explicit memory fallback without consulting browser globals", async () => {
  const archive = await memoryArchive("fallback");
  await archive.save(reading());
  assert.equal(archive.persistence, "memory");
  assert.deepEqual(await archive.list(), [reading()]);
});

test("upsert keeps usable readings and prefers live at equal usability", async () => {
  const archive = await memoryArchive("precedence");
  await archive.save(reading({ reliable: false, mgdl: null, source: "live" }));
  await archive.save(reading({ reliable: true, mgdl: 111, source: "backfill" }));
  await archive.save(reading({ reliable: true, mgdl: 112, source: "live" }));
  assert.deepEqual(await archive.list("sensor-a"), [reading({ reliable: true, mgdl: 112, source: "live" })]);
});

test("bulk ingest is idempotent, supports multiple sensors, and orders chronologically", async () => {
  const archive = await memoryArchive("bulk");
  const rows = [
    reading({ sensorId: "sensor-b", sensorSeconds: 2, atMs: 2_000 }),
    reading({ sensorId: "sensor-a", sensorSeconds: 3, atMs: 1_000 }),
    reading({ sensorId: "sensor-a", sensorSeconds: 1, atMs: 500 }),
  ];
  await archive.ingest(rows);
  await archive.ingest(rows);
  assert.deepEqual(await archive.list(), [rows[2], rows[1], rows[0]]);
  assert.deepEqual(await archive.list("sensor-a"), [rows[2], rows[1]]);
});

test("summaries report each sensor and its latest reading", async () => {
  const archive = await memoryArchive("summary");
  const rows = [
    reading({ sensorId: "sensor-a", sensorSeconds: 1, atMs: 1_000 }),
    reading({ sensorId: "sensor-a", sensorSeconds: 2, atMs: 2_000 }),
    reading({ sensorId: "sensor-b", sensorSeconds: 1, atMs: 500 }),
  ];
  await archive.ingest(rows);
  assert.deepEqual(await archive.summaries(), [
    { sensorId: "sensor-a", count: 2, oldestAtMs: 1_000, newestAtMs: 2_000, latest: rows[1] },
    { sensorId: "sensor-b", count: 1, oldestAtMs: 500, newestAtMs: 500, latest: rows[2] },
  ]);
});

test("skips corrupt rows and supports forgetting one sensor or everything", async () => {
  const indexedDB = new TestIndexedDb();
  const archive = await createReadingArchive({ databaseName: "test-cleanup", indexedDB: indexedDB as unknown as IDBFactory });
  await archive.save(reading({ sensorId: "sensor-a" }));
  await archive.save(reading({ sensorId: "sensor-b" }));
  indexedDB.inject("test-cleanup", { schemaVersion: 1, key: "corrupt", reading: { sensorId: "sensor-a", sensorSeconds: "bad" } });
  assert.equal((await archive.list()).length, 2);
  await archive.forget("sensor-a");
  assert.deepEqual(await archive.list(), [reading({ sensorId: "sensor-b" })]);
  await archive.clear();
  assert.deepEqual(await archive.list(), []);
});
