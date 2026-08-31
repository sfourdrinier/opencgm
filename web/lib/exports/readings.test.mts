import assert from "node:assert/strict";
import test from "node:test";
import { exportReadingsCsv, exportReadingsNdjson, type ExportReading } from "./readings";

const records: ExportReading[] = [
  {
    sensorId: 'sensor,"1"',
    sensorSeconds: 20,
    atMs: Date.UTC(2026, 0, 1, 0, 5),
    receivedAtMs: Date.UTC(2026, 0, 1, 0, 5, 2),
    mgdl: null,
    reliable: false,
    algorithmState: 3,
    source: "backfill",
    credential: "must not export",
    certificateBundle: "must not export",
  },
  {
    sensorId: "sensor-1",
    sensorSeconds: 10,
    atMs: Date.UTC(2026, 0, 1, 0),
    receivedAtMs: Date.UTC(2026, 0, 1, 0, 0, 2),
    mgdl: 111.5,
    reliable: true,
    algorithmState: 2,
    source: "live",
  },
];

test("exports a stable escaped CSV with the documented columns", () => {
  assert.equal(
    exportReadingsCsv(records),
    [
      "timestamp_utc,mgdl,sensor_seconds,source,reliable,algorithm_state,sensor_id",
      '2026-01-01T00:00:00.000Z,111.5,10,live,true,2,sensor-1',
      '2026-01-01T00:05:00.000Z,,20,backfill,false,3,"sensor,""1"""',
      "",
    ].join("\n"),
  );
});

test("emits versioned metadata followed by original records and omits secrets", () => {
  const output = exportReadingsNdjson(records, {
    sensorId: "sensor-1",
    readingCount: records.length,
    duplicateCount: 0,
  });
  const lines = output.trimEnd().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);

  assert.deepEqual(lines[0], {
    version: 1,
    type: "metadata",
    recordCount: 2,
    sensorId: "sensor-1",
    readingCount: 2,
    duplicateCount: 0,
  });
  assert.equal(lines.length, 3);
  assert.deepEqual(lines[1], {
    sensorId: "sensor-1",
    sensorSeconds: 10,
    atMs: Date.UTC(2026, 0, 1, 0),
    receivedAtMs: Date.UTC(2026, 0, 1, 0, 0, 2),
    mgdl: 111.5,
    reliable: true,
    algorithmState: 2,
    source: "live",
  });
  assert.equal(JSON.stringify(lines).includes("credential"), false);
  assert.equal(JSON.stringify(lines).includes("certificate"), false);
});
