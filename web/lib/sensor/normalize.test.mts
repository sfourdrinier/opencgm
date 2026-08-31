import assert from "node:assert/strict";
import test from "node:test";
import { normalizeSensorImport } from "./normalize";
import type { ImportedSensorReading, SensorImportMetadata } from "./contracts";

const metadata: SensorImportMetadata = {
  sensorId: "sensor-a",
  activatedAtMs: 1_000,
  firmware: "1.0",
  oldestAtMs: 2_000,
  newestAtMs: 3_000,
  readingCount: 3,
  duplicateCount: 0,
  historyCompletedThroughSeconds: null,
};

test("retains every provenance record while mapping only finite reliable readings", () => {
  const records: ImportedSensorReading[] = [
    { sensorId: "sensor-a", sensorSeconds: 1, atMs: 2_000, receivedAtMs: 2_100, mgdl: 101, reliable: true, algorithmState: 2, source: "backfill" },
    { sensorId: "sensor-a", sensorSeconds: 2, atMs: 2_300, receivedAtMs: 2_400, mgdl: Number.NaN, reliable: true, algorithmState: 3, source: "live" },
    { sensorId: "sensor-a", sensorSeconds: 3, atMs: 2_600, receivedAtMs: 2_700, mgdl: 102, reliable: false, algorithmState: 4, source: "live" },
  ];

  const result = normalizeSensorImport(records, metadata);

  assert.deepEqual(result.records, records);
  assert.deepEqual(result.analysisReadings, [{ t: 2_000, mgdl: 101 }]);
  assert.deepEqual(result.metadata, metadata);
});

test("marks imports partial and carries warnings without losing records", () => {
  const record: ImportedSensorReading = {
    sensorId: "sensor-a", sensorSeconds: 1, atMs: 2_000, receivedAtMs: 2_100,
    mgdl: null, reliable: false, algorithmState: 9, source: "backfill",
  };

  const result = normalizeSensorImport([record], metadata, "partial", ["history ended early"]);

  assert.equal(result.completeness, "partial");
  assert.deepEqual(result.warnings, ["history ended early"]);
  assert.deepEqual(result.records, [record]);
});
