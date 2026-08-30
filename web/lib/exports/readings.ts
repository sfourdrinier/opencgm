export type ExportReading = {
  readonly sensorId: string;
  readonly sensorSeconds: number;
  readonly atMs: number;
  readonly receivedAtMs: number;
  readonly mgdl: number | null;
  readonly reliable: boolean;
  readonly algorithmState: number;
  readonly source: "live" | "backfill";
  readonly [key: string]: unknown;
};

export type ReadingsExportMetadata = {
  readonly sensorId?: string;
  readonly activatedAtMs?: number | null;
  readonly firmware?: string | null;
  readonly oldestAtMs?: number | null;
  readonly newestAtMs?: number | null;
  readonly readingCount?: number;
  readonly duplicateCount?: number;
  readonly completeness?: "complete" | "partial";
  readonly warnings?: readonly string[];
};

const CSV_COLUMNS = [
  "timestamp_utc",
  "mgdl",
  "sensor_seconds",
  "source",
  "reliable",
  "algorithm_state",
  "sensor_id",
] as const;

function sortedRecords(records: readonly ExportReading[]): ExportReading[] {
  return records
    .map((record, index) => ({ record, index }))
    .sort((a, b) => a.record.atMs - b.record.atMs || a.index - b.index)
    .map(({ record }) => record);
}

function timestamp(atMs: number): string {
  return new Date(atMs).toISOString();
}

function csvField(value: string): string {
  return /[",\n\r]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

function csvValue(value: number | string | boolean | null): string {
  return value === null ? "" : csvField(String(value));
}

export function exportReadingsCsv(records: readonly ExportReading[]): string {
  const rows = [CSV_COLUMNS.join(",")];
  for (const record of sortedRecords(records)) {
    rows.push([
      timestamp(record.atMs),
      csvValue(record.mgdl),
      csvValue(record.sensorSeconds),
      csvValue(record.source),
      csvValue(record.reliable),
      csvValue(record.algorithmState),
      csvValue(record.sensorId),
    ].join(","));
  }
  return `${rows.join("\n")}\n`;
}

export function exportReadingsNdjson(
  records: readonly ExportReading[],
  metadata: ReadingsExportMetadata = {},
): string {
  const sorted = sortedRecords(records);
  const metadataLine: Record<string, unknown> = {
    version: 1,
    type: "metadata",
    recordCount: sorted.length,
  };
  if (metadata.sensorId !== undefined) metadataLine.sensorId = metadata.sensorId;
  if (metadata.activatedAtMs !== undefined) metadataLine.activatedAtMs = metadata.activatedAtMs;
  if (metadata.firmware !== undefined) metadataLine.firmware = metadata.firmware;
  if (metadata.oldestAtMs !== undefined) metadataLine.oldestAtMs = metadata.oldestAtMs;
  if (metadata.newestAtMs !== undefined) metadataLine.newestAtMs = metadata.newestAtMs;
  if (metadata.readingCount !== undefined) metadataLine.readingCount = metadata.readingCount;
  if (metadata.duplicateCount !== undefined) metadataLine.duplicateCount = metadata.duplicateCount;
  if (metadata.completeness !== undefined) metadataLine.completeness = metadata.completeness;
  if (metadata.warnings !== undefined) metadataLine.warnings = metadata.warnings;
  const lines = [JSON.stringify(metadataLine)];
  for (const record of sorted) {
    lines.push(JSON.stringify({
      sensorId: record.sensorId,
      sensorSeconds: record.sensorSeconds,
      atMs: record.atMs,
      receivedAtMs: record.receivedAtMs,
      mgdl: record.mgdl,
      reliable: record.reliable,
      algorithmState: record.algorithmState,
      source: record.source,
    }));
  }
  return `${lines.join("\n")}\n`;
}

export const readingsToCsv = exportReadingsCsv;
export const readingsToNdjson = exportReadingsNdjson;
