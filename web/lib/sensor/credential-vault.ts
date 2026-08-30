const SCHEMA_VERSION = 1;
const KEY_BYTES = 256;
const IV_BYTES = 12;

export type VaultRecord = {
  readonly schemaVersion: number;
  readonly sensorId: string;
  readonly iv: Uint8Array;
  readonly ciphertext: Uint8Array;
  readonly key: CryptoKey;
};

export type VaultDatabase = {
  get(sensorId: string): Promise<VaultRecord | null>;
  put(record: VaultRecord): Promise<void>;
  delete(sensorId: string): Promise<void>;
  clear(): Promise<void>;
};

export type VaultCrypto = {
  generateKey(): Promise<CryptoKey>;
  encrypt(key: CryptoKey, plaintext: Uint8Array, additionalData: Uint8Array, iv: Uint8Array): Promise<Uint8Array>;
  decrypt(key: CryptoKey, ciphertext: Uint8Array, additionalData: Uint8Array, iv: Uint8Array): Promise<Uint8Array>;
  randomBytes(length: number): Uint8Array;
};

export class WebCryptoVaultCrypto implements VaultCrypto {
  readonly #crypto: Crypto;

  constructor(cryptoApi: Crypto = globalThis.crypto) {
    this.#crypto = cryptoApi;
  }

  async generateKey(): Promise<CryptoKey> {
    return this.#crypto.subtle.generateKey({ name: "AES-GCM", length: KEY_BYTES }, false, ["encrypt", "decrypt"]);
  }

  async encrypt(key: CryptoKey, plaintext: Uint8Array, additionalData: Uint8Array, iv: Uint8Array): Promise<Uint8Array> {
    const encrypted = await this.#crypto.subtle.encrypt(
      { name: "AES-GCM", iv: toArrayBuffer(iv), additionalData: toArrayBuffer(additionalData) },
      key,
      toArrayBuffer(plaintext),
    );
    return new Uint8Array(encrypted);
  }

  async decrypt(key: CryptoKey, ciphertext: Uint8Array, additionalData: Uint8Array, iv: Uint8Array): Promise<Uint8Array> {
    const plaintext = await this.#crypto.subtle.decrypt(
      { name: "AES-GCM", iv: toArrayBuffer(iv), additionalData: toArrayBuffer(additionalData) },
      key,
      toArrayBuffer(ciphertext),
    );
    return new Uint8Array(plaintext);
  }

  randomBytes(length: number): Uint8Array {
    return this.#crypto.getRandomValues(new Uint8Array(length));
  }
}

export class MemoryVaultDatabase implements VaultDatabase {
  readonly #records = new Map<string, VaultRecord>();

  async get(sensorId: string): Promise<VaultRecord | null> {
    return this.#records.get(sensorId) ?? null;
  }

  async put(record: VaultRecord): Promise<void> {
    this.#records.set(record.sensorId, record);
  }

  async delete(sensorId: string): Promise<void> {
    this.#records.delete(sensorId);
  }

  async clear(): Promise<void> {
    this.#records.clear();
  }
}

export class IndexedDbVaultDatabase implements VaultDatabase {
  readonly #database: IDBDatabase;

  private constructor(database: IDBDatabase) {
    this.#database = database;
  }

  static async open(factory: IDBFactory, name = "opencgm-credential-vault"): Promise<IndexedDbVaultDatabase> {
    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = factory.open(name, 1);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains("credentials")) request.result.createObjectStore("credentials", { keyPath: "sensorId" });
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
    });
    return new IndexedDbVaultDatabase(database);
  }

  async get(sensorId: string): Promise<VaultRecord | null> {
    return this.request("readonly", (store) => store.get(sensorId));
  }

  async put(record: VaultRecord): Promise<void> {
    await this.request("readwrite", (store) => store.put(record));
  }

  async delete(sensorId: string): Promise<void> {
    await this.request("readwrite", (store) => store.delete(sensorId));
  }

  async clear(): Promise<void> {
    await this.request("readwrite", (store) => store.clear());
  }

  private request<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const transaction = this.#database.transaction("credentials", mode);
      const request = action(transaction.objectStore("credentials"));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
      transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed"));
    });
  }
}

export type CredentialVaultOptions = {
  readonly database?: VaultDatabase;
  readonly crypto?: VaultCrypto;
  readonly persistence?: "memory" | "session-only" | "persistent";
};

export type PersistentCredentialVaultOptions = {
  readonly crypto?: VaultCrypto;
  readonly indexedDB?: IDBFactory;
};

export class CredentialVault {
  readonly database: VaultDatabase;
  readonly persistence: "memory" | "session-only" | "persistent";
  readonly #crypto: VaultCrypto;

  constructor(options: CredentialVaultOptions = {}) {
    this.database = options.database ?? new MemoryVaultDatabase();
    this.persistence = options.persistence ?? (options.database ? "persistent" : "memory");
    this.#crypto = options.crypto ?? new WebCryptoVaultCrypto();
  }

  static async createPersistent(options: PersistentCredentialVaultOptions = {}): Promise<CredentialVault> {
    const factory = options.indexedDB === undefined
      ? (typeof globalThis.indexedDB === "undefined" ? undefined : globalThis.indexedDB)
      : options.indexedDB;
    if (!factory) return new CredentialVault({ crypto: options.crypto, persistence: "session-only" });
    return new CredentialVault({ database: await IndexedDbVaultDatabase.open(factory), crypto: options.crypto });
  }

  static async createPersistentAsync(options: PersistentCredentialVaultOptions = {}): Promise<CredentialVault> {
    return CredentialVault.createPersistent(options);
  }

  async save(sensorId: string, credential: Uint8Array): Promise<void> {
    const key = await this.#crypto.generateKey();
    const iv = this.#crypto.randomBytes(IV_BYTES);
    const ciphertext = await this.#crypto.encrypt(key, credential, associatedData(sensorId), iv);
    await this.database.put({ schemaVersion: SCHEMA_VERSION, sensorId, iv, ciphertext, key });
  }

  async load(sensorId: string): Promise<Uint8Array | null> {
    try {
      const record = await this.database.get(sensorId);
      if (!record || !validRecord(record, sensorId)) return null;
      return await this.#crypto.decrypt(record.key, record.ciphertext, associatedData(sensorId), record.iv);
    } catch {
      return null;
    }
  }

  async forget(sensorId: string): Promise<void> {
    await this.database.delete(sensorId);
  }

  async forgetAll(): Promise<void> {
    await this.database.clear();
  }
}

function associatedData(sensorId: string): Uint8Array {
  return new TextEncoder().encode(`credential-vault:${SCHEMA_VERSION}:${sensorId}`);
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function validRecord(record: VaultRecord, sensorId: string): boolean {
  return record.schemaVersion === SCHEMA_VERSION && record.sensorId === sensorId && record.iv instanceof Uint8Array && record.iv.byteLength === IV_BYTES && record.ciphertext instanceof Uint8Array && record.ciphertext.byteLength > 0 && record.key instanceof CryptoKey;
}
