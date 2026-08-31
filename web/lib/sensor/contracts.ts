// web/lib/sensor/contracts.ts

export const ENGINE_ABI_VERSION = 1;

export type SensorChannel = "authentication" | "control" | "backfill" | "extra-data";
export type TerminalReason = "link-lost" | "page-hidden" | "cancelled" | "operation-failed";
export type PublicSensorError =
  | "authentication-rejected"
  | "pairing-required"
  | "credential-invalid"
  | "connection-ended"
  | "history-incomplete"
  | "protocol-invalid"
  | "cancelled";

export type ImportedSensorReading = {
  readonly sensorId: string;
  readonly sensorSeconds: number;
  readonly atMs: number;
  readonly receivedAtMs: number;
  readonly mgdl: number | null;
  readonly reliable: boolean;
  readonly algorithmState: number;
  readonly source: "live" | "backfill";
};

export type SensorImportMetadata = {
  readonly sensorId: string;
  readonly activatedAtMs: number | null;
  readonly firmware: string | null;
  readonly oldestAtMs: number | null;
  readonly newestAtMs: number | null;
  readonly readingCount: number;
  readonly duplicateCount: number;
  readonly historyCompletedThroughSeconds: number | null;
};

export type SensorImportResult = {
  readonly records: readonly ImportedSensorReading[];
  readonly analysisReadings: readonly { readonly t: number; readonly mgdl: number }[];
  readonly metadata: SensorImportMetadata;
  readonly completeness: "complete" | "partial";
  readonly warnings: readonly string[];
};

export type EngineCommand =
  | { readonly kind: "start"; readonly nowMs: number; readonly sensorName: string; readonly credential: Uint8Array | null; readonly pairingCode: string | null; readonly certificateBundle: Uint8Array | null; readonly historyStartSeconds?: number | null }
  | { readonly kind: "frame"; readonly channel: SensorChannel; readonly bytes: Uint8Array; readonly nowMs: number }
  | { readonly kind: "action-result"; readonly actionId: number; readonly ok: boolean; readonly bytes: Uint8Array }
  | { readonly kind: "terminal"; readonly reason: TerminalReason }
  | { readonly kind: "stop" };

export type EngineAction =
  | { readonly kind: "need-entropy"; readonly actionId: number; readonly byteCount: number }
  | { readonly kind: "subscribe"; readonly actionId: number; readonly channel: SensorChannel }
  | { readonly kind: "write"; readonly actionId: number; readonly channel: SensorChannel; readonly response: "required" | "not-required"; readonly bytes: Uint8Array; readonly delayAfterMs: number }
  | { readonly kind: "request-os-pair"; readonly actionId: number }
  | { readonly kind: "persist-credential"; readonly credential: Uint8Array }
  | { readonly kind: "reading"; readonly reading: ImportedSensorReading }
  | { readonly kind: "metadata"; readonly metadata: SensorImportMetadata }
  | { readonly kind: "complete"; readonly completeness: "complete" | "partial" }
  | { readonly kind: "failure"; readonly category: PublicSensorError };

export class EngineAbiError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "EngineAbiError";
  }
}

export const ENGINE_ABI_MAX_BYTES = 1024 * 1024;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8", { fatal: true });
const channels: readonly SensorChannel[] = ["authentication", "control", "backfill", "extra-data"];
const terminalReasons: readonly TerminalReason[] = ["link-lost", "page-hidden", "cancelled", "operation-failed"];
const publicErrors: readonly PublicSensorError[] = [
  "authentication-rejected", "pairing-required", "credential-invalid", "connection-ended",
  "history-incomplete", "protocol-invalid", "cancelled",
];

function enumCode<T extends string>(values: readonly T[], value: T, name: string): number {
  const code = values.indexOf(value);
  if (code < 0) throw new EngineAbiError(`Unknown ${name}`);
  return code;
}

function enumValue<T extends string>(values: readonly T[], code: number, name: string): T {
  const value = values[code];
  if (value === undefined) throw new EngineAbiError(`Unknown ${name} tag`);
  return value;
}

function assertSafeU64(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) throw new EngineAbiError(`Invalid ${name}`);
}

function assertU32(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) throw new EngineAbiError(`Invalid ${name}`);
}

function assertI32(value: number, name: string): void {
  if (!Number.isInteger(value) || value < -0x80000000 || value > 0x7fffffff) throw new EngineAbiError(`Invalid ${name}`);
}

class Writer {
  private readonly bytes: number[] = [];
  public u8(value: number): void {
    if (!Number.isInteger(value) || value < 0 || value > 0xff) throw new EngineAbiError("Invalid u8");
    this.bytes.push(value);
  }
  public u32(value: number): void {
    assertU32(value, "u32");
    this.bytes.push(value & 0xff, (value >>> 8) & 0xff, (value >>> 16) & 0xff, (value >>> 24) & 0xff);
  }
  public u64(value: number): void {
    assertSafeU64(value, "u64");
    const view = new DataView(new ArrayBuffer(8));
    view.setBigUint64(0, BigInt(value), true);
    for (const byte of new Uint8Array(view.buffer)) this.bytes.push(byte);
  }
  public i32(value: number): void {
    assertI32(value, "i32");
    const view = new DataView(new ArrayBuffer(4));
    view.setInt32(0, value, true);
    for (const byte of new Uint8Array(view.buffer)) this.bytes.push(byte);
  }
  public bytesValue(value: Uint8Array): void {
    if (value.length > ENGINE_ABI_MAX_BYTES) throw new EngineAbiError("Payload is too large");
    this.u32(value.length);
    for (const byte of value) this.bytes.push(byte);
  }
  public string(value: string): void { this.bytesValue(textEncoder.encode(value)); }
  public optionalBytes(value: Uint8Array | null): void { this.u8(value === null ? 0 : 1); if (value !== null) this.bytesValue(value); }
  public optionalString(value: string | null): void { this.u8(value === null ? 0 : 1); if (value !== null) this.string(value); }
  public optionalU64(value: number | null): void { this.u8(value === null ? 0 : 1); if (value !== null) this.u64(value); }
  public optionalU32(value: number | null): void { this.u8(value === null ? 0 : 1); if (value !== null) this.u32(value); }
  public finish(): Uint8Array {
    if (this.bytes.length > ENGINE_ABI_MAX_BYTES) throw new EngineAbiError("Payload is too large");
    return Uint8Array.from(this.bytes);
  }
}

class Reader {
  private offset = 0;
  public constructor(private readonly bytes: Uint8Array) {
    if (bytes.length > ENGINE_ABI_MAX_BYTES) throw new EngineAbiError("Payload is too large");
  }
  private take(length: number): Uint8Array {
    if (!Number.isSafeInteger(length) || length < 0 || this.offset + length > this.bytes.length) throw new EngineAbiError("Truncated payload");
    const result = this.bytes.slice(this.offset, this.offset + length);
    this.offset += length;
    return result;
  }
  public u8(): number { return this.take(1)[0] ?? this.fail("Truncated u8"); }
  public u32(): number {
    const bytes = this.take(4);
    return (((bytes[0] ?? 0) | ((bytes[1] ?? 0) << 8) | ((bytes[2] ?? 0) << 16) | ((bytes[3] ?? 0) << 24)) >>> 0);
  }
  public u64(): number {
    const bytes = this.take(8);
    const value = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(0, true);
    const number = Number(value);
    if (!Number.isSafeInteger(number)) throw new EngineAbiError("Integer exceeds safe range");
    return number;
  }
  public i32(): number { const bytes = this.take(4); return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getInt32(0, true); }
  public bytesValue(): Uint8Array { const length = this.u32(); if (length > ENGINE_ABI_MAX_BYTES) throw new EngineAbiError("Payload is too large"); return this.take(length); }
  public string(): string {
    try { return textDecoder.decode(this.bytesValue()); }
    catch (error) { throw new EngineAbiError(`Invalid UTF-8 string: ${error instanceof Error ? error.message : "decode failed"}`); }
  }
  public optionalBytes(): Uint8Array | null { const present = this.u8(); if (present === 0) return null; if (present !== 1) throw new EngineAbiError("Invalid optional value tag"); return this.bytesValue(); }
  public optionalString(): string | null { const present = this.u8(); if (present === 0) return null; if (present !== 1) throw new EngineAbiError("Invalid optional value tag"); return this.string(); }
  public optionalU64(): number | null { const present = this.u8(); if (present === 0) return null; if (present !== 1) throw new EngineAbiError("Invalid optional value tag"); return this.u64(); }
  public optionalU32(): number | null { const present = this.u8(); if (present === 0) return null; if (present !== 1) throw new EngineAbiError("Invalid optional value tag"); return this.u32(); }
  public hasRemaining(): boolean { return this.offset < this.bytes.length; }
  public done(): void { if (this.offset !== this.bytes.length) throw new EngineAbiError("Trailing bytes"); }
  private fail(message: string): never { throw new EngineAbiError(message); }
}

function writeReading(writer: Writer, reading: ImportedSensorReading): void {
  writer.string(reading.sensorId); writer.u32(reading.sensorSeconds); writer.u64(reading.atMs); writer.u64(reading.receivedAtMs);
  writer.u8(reading.mgdl === null ? 0 : 1); if (reading.mgdl !== null) writer.i32(reading.mgdl);
  writer.u8(reading.reliable ? 1 : 0); writer.u32(reading.algorithmState); writer.u8(reading.source === "live" ? 0 : 1);
}

function readReading(reader: Reader): ImportedSensorReading {
  const sensorId = reader.string(); const sensorSeconds = reader.u32(); const atMs = reader.u64(); const receivedAtMs = reader.u64();
  const hasMgdl = reader.u8(); if (hasMgdl !== 0 && hasMgdl !== 1) throw new EngineAbiError("Invalid glucose presence tag");
  const mgdl = hasMgdl === 1 ? reader.i32() : null; const reliableTag = reader.u8();
  if (reliableTag !== 0 && reliableTag !== 1) throw new EngineAbiError("Invalid reliable flag");
  const algorithmState = reader.u32(); const source = enumValue(["live", "backfill"] as const, reader.u8(), "reading source");
  return { sensorId, sensorSeconds, atMs, receivedAtMs, mgdl, reliable: reliableTag === 1, algorithmState, source };
}

function writeMetadata(writer: Writer, metadata: SensorImportMetadata): void {
  writer.string(metadata.sensorId); writer.optionalU64(metadata.activatedAtMs); writer.optionalString(metadata.firmware);
  writer.optionalU64(metadata.oldestAtMs); writer.optionalU64(metadata.newestAtMs); writer.u32(metadata.readingCount); writer.u32(metadata.duplicateCount); writer.optionalU32(metadata.historyCompletedThroughSeconds);
}
function readMetadata(reader: Reader): SensorImportMetadata {
  return { sensorId: reader.string(), activatedAtMs: reader.optionalU64(), firmware: reader.optionalString(), oldestAtMs: reader.optionalU64(), newestAtMs: reader.optionalU64(), readingCount: reader.u32(), duplicateCount: reader.u32(), historyCompletedThroughSeconds: reader.hasRemaining() ? reader.optionalU32() : null };
}
function finishDecode<T>(reader: Reader, value: T): T { reader.done(); return value; }

export function encodeEngineCommand(command: EngineCommand): Uint8Array {
  const writer = new Writer(); writer.u8(ENGINE_ABI_VERSION);
  switch (command.kind) {
    case "start": writer.u8(0); writer.u64(command.nowMs); writer.string(command.sensorName); writer.optionalBytes(command.credential); writer.optionalString(command.pairingCode); writer.optionalBytes(command.certificateBundle); writer.optionalU32(command.historyStartSeconds ?? null); break;
    case "frame": writer.u8(1); writer.u8(enumCode(channels, command.channel, "channel")); writer.bytesValue(command.bytes); writer.u64(command.nowMs); break;
    case "action-result": writer.u8(2); writer.u32(command.actionId); writer.u8(command.ok ? 1 : 0); writer.bytesValue(command.bytes); break;
    case "terminal": writer.u8(3); writer.u8(enumCode(terminalReasons, command.reason, "terminal reason")); break;
    case "stop": writer.u8(4); break;
  }
  return writer.finish();
}

export function decodeEngineCommand(bytes: Uint8Array): EngineCommand {
  const reader = new Reader(bytes); if (reader.u8() !== ENGINE_ABI_VERSION) throw new EngineAbiError("Unsupported ABI version");
  switch (reader.u8()) {
    case 0: {
      const nowMs = reader.u64(); const sensorName = reader.string(); const credential = reader.optionalBytes(); const pairingCode = reader.optionalString(); const certificateBundle = reader.optionalBytes();
      const historyStartSeconds = reader.hasRemaining() ? reader.optionalU32() : null;
      return finishDecode(reader, { kind: "start", nowMs, sensorName, credential, pairingCode, certificateBundle, historyStartSeconds });
    }
    case 1: return finishDecode(reader, { kind: "frame", channel: enumValue(channels, reader.u8(), "channel"), bytes: reader.bytesValue(), nowMs: reader.u64() });
    case 2: { const actionId = reader.u32(); const okTag = reader.u8(); if (okTag !== 0 && okTag !== 1) throw new EngineAbiError("Invalid action result flag"); return finishDecode(reader, { kind: "action-result", actionId, ok: okTag === 1, bytes: reader.bytesValue() }); }
    case 3: return finishDecode(reader, { kind: "terminal", reason: enumValue(terminalReasons, reader.u8(), "terminal reason") });
    case 4: return finishDecode(reader, { kind: "stop" });
    default: throw new EngineAbiError("Unknown command tag");
  }
}

export function encodeEngineAction(action: EngineAction): Uint8Array {
  const writer = new Writer(); writer.u8(ENGINE_ABI_VERSION);
  switch (action.kind) {
    case "need-entropy": writer.u8(0); writer.u32(action.actionId); writer.u32(action.byteCount); break;
    case "subscribe": writer.u8(1); writer.u32(action.actionId); writer.u8(enumCode(channels, action.channel, "channel")); break;
    case "write": writer.u8(2); writer.u32(action.actionId); writer.u8(enumCode(channels, action.channel, "channel")); writer.u8(action.response === "required" ? 1 : 0); writer.bytesValue(action.bytes); writer.u32(action.delayAfterMs); break;
    case "request-os-pair": writer.u8(3); writer.u32(action.actionId); break;
    case "persist-credential": writer.u8(4); writer.bytesValue(action.credential); break;
    case "reading": writer.u8(5); writeReading(writer, action.reading); break;
    case "metadata": writer.u8(6); writeMetadata(writer, action.metadata); break;
    case "complete": writer.u8(7); writer.u8(action.completeness === "complete" ? 0 : 1); break;
    case "failure": writer.u8(8); writer.u8(enumCode(publicErrors, action.category, "failure category")); break;
  }
  return writer.finish();
}

export function decodeEngineAction(bytes: Uint8Array): EngineAction {
  const reader = new Reader(bytes); if (reader.u8() !== ENGINE_ABI_VERSION) throw new EngineAbiError("Unsupported ABI version");
  switch (reader.u8()) {
    case 0: return finishDecode(reader, { kind: "need-entropy", actionId: reader.u32(), byteCount: reader.u32() });
    case 1: return finishDecode(reader, { kind: "subscribe", actionId: reader.u32(), channel: enumValue(channels, reader.u8(), "channel") });
    case 2: { const actionId = reader.u32(); const channel = enumValue(channels, reader.u8(), "channel"); const responseTag = reader.u8(); if (responseTag !== 0 && responseTag !== 1) throw new EngineAbiError("Invalid response flag"); return finishDecode(reader, { kind: "write", actionId, channel, response: responseTag === 1 ? "required" : "not-required", bytes: reader.bytesValue(), delayAfterMs: reader.u32() }); }
    case 3: return finishDecode(reader, { kind: "request-os-pair", actionId: reader.u32() });
    case 4: return finishDecode(reader, { kind: "persist-credential", credential: reader.bytesValue() });
    case 5: return finishDecode(reader, { kind: "reading", reading: readReading(reader) });
    case 6: return finishDecode(reader, { kind: "metadata", metadata: readMetadata(reader) });
    case 7: { const completeness = reader.u8(); if (completeness !== 0 && completeness !== 1) throw new EngineAbiError("Invalid completeness tag"); return finishDecode(reader, { kind: "complete", completeness: completeness === 0 ? "complete" : "partial" }); }
    case 8: return finishDecode(reader, { kind: "failure", category: enumValue(publicErrors, reader.u8(), "failure category") });
    default: throw new EngineAbiError("Unknown action tag");
  }
}

export function decodeEngineActions(bytes: Uint8Array): EngineAction[] {
  // ABI 1 artifacts may return one action directly; current engines return
  // the same action envelopes in a bounded list. Accept both wire envelopes
  // while keeping action decoding strict and protocol-agnostic.
  if (bytes.length < 5) return [decodeEngineAction(bytes)];
  try {
    const reader = new Reader(bytes); if (reader.u8() !== ENGINE_ABI_VERSION) throw new EngineAbiError("Unsupported ABI version");
    const count = reader.u32(); const actions: EngineAction[] = []; if (count > Math.floor((bytes.length - 5) / 4)) throw new EngineAbiError("Too many actions");
    for (let index = 0; index < count; index += 1) actions.push(decodeEngineAction(reader.bytesValue()));
    reader.done(); return actions;
  } catch (error) {
    if (error instanceof EngineAbiError) return [decodeEngineAction(bytes)];
    throw error;
  }
}
