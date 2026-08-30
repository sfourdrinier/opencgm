import assert from "node:assert/strict";
import test from "node:test";
import type { SensorImportResult } from "../lib/sensor/contracts";
import {
  browserSupportMessage,
  importProgressLabel,
  privacyCopy,
  sensorImportDiagnosticsVisible,
  sensorImportCoverage,
} from "./SensorImportClient";

const imported: SensorImportResult = {
  records: [
    {
      sensorId: "g7-demo",
      sensorSeconds: 10,
      atMs: Date.UTC(2026, 7, 28, 8),
      receivedAtMs: Date.UTC(2026, 7, 28, 8, 0, 1),
      mgdl: 101,
      reliable: true,
      algorithmState: 1,
      source: "backfill",
    },
  ],
  analysisReadings: [{ t: Date.UTC(2026, 7, 28, 8), mgdl: 101 }],
  metadata: {
    sensorId: "g7-demo",
    activatedAtMs: null,
    firmware: null,
    oldestAtMs: Date.UTC(2026, 7, 28, 8),
    newestAtMs: Date.UTC(2026, 7, 28, 8),
    readingCount: 1,
    duplicateCount: 2,
  },
  completeness: "partial",
  warnings: ["History is partial."],
};

test("explains Chrome, sensor, Bluetooth, and origin prerequisites without a UA gate", () => {
  assert.match(browserSupportMessage({ secureContext: false, bluetooth: false }), /Chrome/);
  assert.match(browserSupportMessage({ secureContext: false, bluetooth: false }), /Dexcom G7/);
  assert.match(browserSupportMessage({ secureContext: false, bluetooth: false }), /Bluetooth/);
  assert.doesNotMatch(browserSupportMessage({ secureContext: false, bluetooth: false }), /HTTPS|localhost/);
  assert.match(browserSupportMessage({ secureContext: true, bluetooth: true }), /ready/i);
  assert.match(browserSupportMessage({ secureContext: false, bluetooth: true, origin: "http://localhost:3000" }), /ready/i);
});

test("summarises counts, duplicate rows, range, cadence, gaps, and completeness", () => {
  const summary = sensorImportCoverage(imported);
  assert.equal(summary.readings, 1);
  assert.equal(summary.duplicates, 2);
  assert.equal(summary.completeness, "partial");
  assert.equal(summary.range, "28 Aug 2026, 08:00–08:00");
  assert.equal(summary.cadence, "one reading");
  assert.match(summary.gaps, /No interval/);
});

test("uses user-facing progress labels and never exposes protocol internals", () => {
  assert.equal(importProgressLabel("loading"), "Preparing a local connection…");
  assert.equal(importProgressLabel("connecting"), "Connect your Dexcom G7 when prompted…");
  assert.equal(importProgressLabel("backfill"), "Reading available history…");
  assert.equal(importProgressLabel("analysis"), "Running the full analysis…");
  assert.equal(importProgressLabel("done"), "Import ready");
});

test("hides diagnostic credential inputs on the normal route", () => {
  assert.equal(sensorImportDiagnosticsVisible(""), false);
  assert.equal(sensorImportDiagnosticsVisible("?debug=no"), false);
  assert.equal(sensorImportDiagnosticsVisible("?debug=yesplease&other=1"), false);
});

test("reveals diagnostic credential inputs only for the exact debug query", () => {
  assert.equal(sensorImportDiagnosticsVisible("?debug=yesplease"), true);
  assert.equal(sensorImportDiagnosticsVisible("?debug=yespleasex"), false);
  assert.equal(sensorImportDiagnosticsVisible("?x=1&debug=yesplease"), false);
});

test("uses the exact prominent local-processing privacy copy", () => {
  assert.equal(privacyCopy, "Your data stays in this browser. Sensor readings and credentials are processed locally and are never uploaded or sent to a backend.");
});
