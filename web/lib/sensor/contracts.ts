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
};

export type SensorImportResult = {
  readonly records: readonly ImportedSensorReading[];
  readonly analysisReadings: readonly { readonly t: number; readonly mgdl: number }[];
  readonly metadata: SensorImportMetadata;
  readonly completeness: "complete" | "partial";
  readonly warnings: readonly string[];
};

export type EngineCommand =
  | { readonly kind: "start"; readonly nowMs: number; readonly sensorName: string; readonly credential: Uint8Array | null; readonly pairingCode: string | null; readonly certificateBundle: Uint8Array | null }
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
