import test from "node:test";
import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import {
  CredentialVault,
  MemoryVaultDatabase,
  WebCryptoVaultCrypto,
  type VaultDatabase,
  type VaultRecord,
} from "./credential-vault";

const cryptoApi = webcrypto as unknown as Crypto;

test("encrypts and decrypts an opaque credential with fresh IVs and sensor-scoped replacement", async () => {
  const database = new MemoryVaultDatabase();
  const vault = new CredentialVault({ database, crypto: new WebCryptoVaultCrypto(cryptoApi) });
  const first = new Uint8Array([1, 2, 3]);
  const replacement = new Uint8Array([4, 5]);

  await vault.save("sensor-a", first);
  const firstRecord = await database.get("sensor-a");
  assert.deepEqual(await vault.load("sensor-a"), first);
  await vault.save("sensor-a", replacement);
  const secondRecord = await database.get("sensor-a");
  assert.deepEqual(await vault.load("sensor-a"), replacement);
  assert.notDeepEqual(firstRecord?.iv, secondRecord?.iv);
  assert.equal(await vault.load("sensor-b"), null);
  assert.notDeepEqual(firstRecord?.ciphertext, secondRecord?.ciphertext);
});

test("uses memory-only storage by default and forgets one or all sensors", async () => {
  const vault = new CredentialVault({ crypto: new WebCryptoVaultCrypto(cryptoApi) });
  await vault.save("a", new Uint8Array([7]));
  await vault.save("b", new Uint8Array([8]));
  await vault.forget("a");
  assert.equal(await vault.load("a"), null);
  assert.deepEqual(await vault.load("b"), new Uint8Array([8]));
  await vault.forgetAll();
  assert.equal(await vault.load("b"), null);
  assert.equal(vault.persistence, "memory");
});

test("fails closed for wrong key, corrupt ciphertext, and unsupported schema", async () => {
  const database = new MemoryVaultDatabase();
  const vault = new CredentialVault({ database, crypto: new WebCryptoVaultCrypto(cryptoApi) });
  await vault.save("sensor", new Uint8Array([9, 10]));
  const record = await database.get("sensor");
  assert.ok(record);
  await database.put({ ...record, ciphertext: new Uint8Array(record.ciphertext).fill(0) });
  assert.equal(await vault.load("sensor"), null);
  await database.put({ ...record, schemaVersion: 999 });
  assert.equal(await vault.load("sensor"), null);

  const other = new CredentialVault({ database: new MemoryVaultDatabase(), crypto: new WebCryptoVaultCrypto(cryptoApi) });
  await other.save("sensor", new Uint8Array([9, 10]));
  const otherRecord = await other.database.get("sensor");
  assert.ok(otherRecord);
  await database.put({ ...record, key: otherRecord.key });
  assert.equal(await vault.load("sensor"), null);
});

test("reports explicit session-only mode when persistent storage is unavailable", async () => {
  const vault = await CredentialVault.createPersistent({
    crypto: new WebCryptoVaultCrypto(cryptoApi),
    indexedDB: undefined,
  });
  assert.equal(vault.persistence, "session-only");
  await vault.save("sensor", new Uint8Array([1]));
  assert.deepEqual(await vault.load("sensor"), new Uint8Array([1]));
});

test("database values contain only encrypted opaque credential material", async () => {
  const database = new MemoryVaultDatabase();
  const vault = new CredentialVault({ database, crypto: new WebCryptoVaultCrypto(cryptoApi) });
  await vault.save("sensor", new Uint8Array([1, 2, 3]));
  const record: VaultRecord | null = await database.get("sensor");
  assert.ok(record);
  assert.deepEqual(Object.keys(record).sort(), ["ciphertext", "iv", "key", "schemaVersion", "sensorId"]);
  assert.equal(record.sensorId, "sensor");
  assert.notDeepEqual(record.ciphertext, new Uint8Array([1, 2, 3]));
});

test("injected database boundary can be used without persistence assumptions", async () => {
  const entries = new Map<string, VaultRecord>();
  const database: VaultDatabase = {
    get: async (id) => entries.get(id) ?? null,
    put: async (record) => void entries.set(record.sensorId, record),
    delete: async (id) => void entries.delete(id),
    clear: async () => void entries.clear(),
  };
  const vault = new CredentialVault({ database, crypto: new WebCryptoVaultCrypto(cryptoApi) });
  await vault.save("sensor", new Uint8Array([11]));
  assert.deepEqual(await vault.load("sensor"), new Uint8Array([11]));
});
