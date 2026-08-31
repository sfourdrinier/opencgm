import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { BleError } from "unified-ble-manager";
import type { SensorImportResult } from "../lib/sensor/contracts";
import {
  browserCapabilitySnapshot,
  browserSupportMessage,
  isValidPairingCode,
  importProgressLabel,
  privacyCopy,
  pairingCodeGuidance,
  pairingCodeRequiredCopy,
  sensorImportCredentialCallbacks,
  sensorImportBluetoothOptions,
  sensorImportDiagnosticsVisible,
  sensorImportCoverage,
  sensorImportWaitingCopy,
  sensorSelectedCopy,
  sensorImportError,
  sensorImportDiagnostic,
  SensorImportClient,
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
    historyCompletedThroughSeconds: null,
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

test("recomputes localhost Bluetooth capability after client mount", () => {
  const capability = browserCapabilitySnapshot({ secureContext: false, navigator: { bluetooth: {} }, origin: "http://localhost:3000" });
  assert.deepEqual(capability, { secureContext: false, bluetooth: true, origin: "http://localhost:3000" });
  assert.match(browserSupportMessage(capability), /ready/i);
});

test("gives Linux Chrome an exact adapter, flag, relaunch, and reload recovery", () => {
  const message = browserSupportMessage({ secureContext: true, bluetooth: false, userAgent: "Mozilla/5.0 (X11; Linux x86_64) Chrome/140.0" });
  assert.match(message, /hci0/);
  assert.match(message, /--enable-experimental-web-platform-features/);
  assert.match(message, /relaunch Chrome/);
  assert.match(message, /reload this page/);
});

test("keeps non-Linux Chrome recovery guidance generic", () => {
  const message = browserSupportMessage({ secureContext: true, bluetooth: false, userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/140.0" });
  assert.match(message, /Google Chrome/);
  assert.doesNotMatch(message, /hci0|enable-experimental-web-platform-features/);
  const androidMessage = browserSupportMessage({ secureContext: true, bluetooth: false, userAgent: "Mozilla/5.0 (Linux; Android 15) Chrome/140.0" });
  assert.doesNotMatch(androidMessage, /hci0|enable-experimental-web-platform-features/);
});

test("requires exactly four ASCII digits for a first pairing", () => {
  assert.equal(pairingCodeRequiredCopy, "— Need pairing code");
  assert.equal(isValidPairingCode("1234"), true);
  assert.equal(isValidPairingCode(""), false);
  assert.equal(isValidPairingCode("123"), false);
  assert.equal(isValidPairingCode("12345"), false);
  assert.equal(isValidPairingCode("１２３４"), false);
  assert.match(pairingCodeGuidance, /four ASCII digits/);
});

test("uses truthful waiting copy for intermittent sensor availability", () => {
  assert.match(sensorImportWaitingCopy, /about every five minutes/);
  assert.match(sensorImportWaitingCopy, /keep this page open/i);
  assert.match(sensorImportWaitingCopy, /several minutes/);
  assert.doesNotMatch(sensorImportWaitingCopy, /exactly|seconds|window/);
});

test("explains that a selected sensor is connecting instead of asking for another prompt", () => {
  assert.equal(sensorSelectedCopy, "Sensor selected. Waiting up to five minutes for its Bluetooth connection…");
});

test("keeps one selected-device connection attempt alive for the sensor wake interval", () => {
  assert.equal(sensorImportBluetoothOptions.connectTimeoutMs, 300_000);
  assert.equal(sensorImportBluetoothOptions.maxAttempts, 25);
  assert.equal(sensorImportBluetoothOptions.retryDelayMs, 15_000);
});

test("describes Chrome NotFoundError without claiming the user cancelled", () => {
  const error = new BleError("chooser.cancelled", "chooser", "web-chooser.choose", {
    platform: {
      domain: "web-bluetooth",
      code: "NotFoundError",
      safeMessage: "The Web Bluetooth operation failed.",
      metadata: { browserErrorName: "NotFoundError" },
    },
  });
  assert.equal(
    sensorImportError(error),
    "The Bluetooth chooser closed without selecting a compatible sensor. Keep the sensor nearby and try again.",
  );
});

test("explains the browser OS-bond boundary without claiming first-pair success", () => {
  const message = sensorImportError(new Error("browser Web Bluetooth does not support OS pairing"));
  assert.match(message, /cannot finish a new operating-system Bluetooth bond/i);
  assert.match(message, /browser-authorized sensor/i);
});

test("turns a connection deadline into actionable retry guidance", () => {
  assert.equal(
    sensorImportError(new Error("operation timed out")),
    "The sensor connection timed out. Keep the sensor nearby, stop any competing phone connection, and retry during its next Bluetooth window.",
  );
});

test("preserves structured Web Bluetooth connection evidence for users and support", () => {
  const error = new BleError("connection.failed", "connection", "web-connection.connect", {
    platform: {
      domain: "web-bluetooth",
      code: "NetworkError",
      safeMessage: "The Web Bluetooth operation failed.",
      metadata: { browserErrorName: "NetworkError" },
    },
  });
  assert.equal(
    sensorImportError(error),
    "Chrome found the sensor but could not open its Bluetooth connection. Stop any competing phone connection, wait for the next sensor window, and retry.",
  );
  assert.equal(sensorImportDiagnostic(error), "connection.failed · web-connection.connect · NetworkError");
});

test("uses the proven Dexcom advertisement and GATT UUIDs", () => {
  assert.equal(sensorImportBluetoothOptions.connectTimeoutMs, 300_000);
  assert.equal(sensorImportBluetoothOptions.chooserServiceUuid, "0000febc-0000-1000-8000-00805f9b34fb");
  assert.equal(sensorImportBluetoothOptions.serviceUuid, "f8083532-849e-531c-c594-30f1f86a4ea5");
  assert.deepEqual(sensorImportBluetoothOptions.channels, {
    authentication: "f8083535-849e-531c-c594-30f1f86a4ea5",
    control: "f8083534-849e-531c-c594-30f1f86a4ea5",
    backfill: "f8083536-849e-531c-c594-30f1f86a4ea5",
    "extra-data": "f8083538-849e-531c-c594-30f1f86a4ea5",
  });
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

test("keeps the first debug render identical to server HTML", () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "window");
  Reflect.deleteProperty(globalThis, "window");
  const serverHtml = renderToStaticMarkup(createElement(SensorImportClient));

  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { location: { search: "?debug=yesplease" } },
  });
  try {
    const firstBrowserHtml = renderToStaticMarkup(createElement(SensorImportClient));
    assert.equal(firstBrowserHtml, serverHtml);
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "window", descriptor);
    else Reflect.deleteProperty(globalThis, "window");
  }
});

test("uses the exact prominent local-processing privacy copy", () => {
  assert.equal(privacyCopy, "Your data stays in this browser. Sensor readings and credentials are processed locally and are never uploaded or sent to a backend.");
});

test("wires vault callbacks to the selected peer and keeps saving opt-in", async () => {
  const loads: string[] = [];
  const saves: Array<{ peerId: string; credential: Uint8Array }> = [];
  const selected: string[] = [];
  const vault = {
    load: async (peerId: string) => { loads.push(peerId); return new Uint8Array([4]); },
    save: async (peerId: string, credential: Uint8Array) => { saves.push({ peerId, credential }); },
  };
  const persistedReadings: string[] = [];
  const callbacks = sensorImportCredentialCallbacks(
    vault,
    true,
    peerId => selected.push(peerId),
    reading => { persistedReadings.push(`${reading.sensorId}:${reading.sensorSeconds}`); },
  );

  callbacks.onPeerSelected?.("stable-peer");
  const loaded = await callbacks.loadCredential?.("stable-peer");
  await callbacks.saveCredential?.("stable-peer", new Uint8Array([5]));
  await callbacks.onReading?.(imported.records[0]!);

  assert.deepEqual(selected, ["stable-peer"]);
  assert.deepEqual(loads, ["stable-peer"]);
  assert.deepEqual(loaded, new Uint8Array([4]));
  assert.deepEqual(saves, [{ peerId: "stable-peer", credential: new Uint8Array([5]) }]);
  assert.deepEqual(persistedReadings, ["g7-demo:10"]);
  assert.equal(sensorImportCredentialCallbacks(vault, false, () => undefined).saveCredential, undefined);
});
