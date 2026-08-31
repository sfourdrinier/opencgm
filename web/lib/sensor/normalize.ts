import type { Reading } from "../types";
import type {
  ImportedSensorReading,
  SensorImportMetadata,
  SensorImportResult,
} from "./contracts";

/**
 * Preserve the wire history as-is while exposing the conservative subset that
 * the existing analysis pipeline can safely consume.
 */
export function normalizeSensorImport(
  records: readonly ImportedSensorReading[],
  metadata: SensorImportMetadata,
  completeness: SensorImportResult["completeness"] = "complete",
  warnings: readonly string[] = [],
): SensorImportResult {
  const analysisReadings: Reading[] = [];
  for (const record of records) {
    if (!record.reliable || record.mgdl === null) continue;
    if (!Number.isFinite(record.atMs) || !Number.isFinite(record.mgdl)) continue;
    analysisReadings.push({ t: record.atMs, mgdl: record.mgdl });
  }
  return {
    records: Object.freeze([...records]),
    analysisReadings: Object.freeze(analysisReadings),
    metadata,
    completeness,
    warnings: Object.freeze([...warnings]),
  };
}

/** Narrow alias for callers that already hold a history list. */
export const normalizeImportedReadings = normalizeSensorImport;
