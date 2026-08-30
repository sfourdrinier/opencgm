// Regression tests for CSV column detection.
//
// Run with: npm run test
//
// The fixture below is shaped like a Heart Studio export of a Dexcom sensor: it carries both
// `sensor_timestamp_seconds` (a monotonic counter that starts near zero at sensor activation)
// and `observed_at_utc` (the wall clock). Detection that matches on header names alone picks
// the counter, because it contains the word "timestamp" and comes first -- and then every
// reading lands in 1970 with nothing appearing to fail.
//
// The values are invented. Real exports are personal health data and do not belong in a
// public repository, even as a test fixture.

import assert from "node:assert/strict";
import Papa from "papaparse";
import { hasExplicitOffset, pickTimestampColumn } from "./parse";

function rows(csv: string) {
  const out = Papa.parse<Record<string, string>>(csv, { header: true, skipEmptyLines: true });
  return { header: out.meta.fields ?? [], data: out.data };
}

let failures = 0;
function test(name: string, fn: () => void) {
  try {
    fn();
    console.log(`  ok   ${name}`);
  } catch (e) {
    failures += 1;
    console.error(`  FAIL ${name}\n       ${e instanceof Error ? e.message : String(e)}`);
  }
}

console.log("csv column detection");

const HEARTSTUDIO = `sensor_id,sensor_timestamp_seconds,observed_at_utc,glucose_mg_dl,algorithm_state,reliable
s-1,477,2026-08-25T18:43:00.739Z,95.0,.known(.warmup),0
s-1,777,2026-08-25T18:48:00.739Z,92.0,.known(.ok),1
s-1,1077,2026-08-25T18:53:00.739Z,97.0,.known(.ok),1`;

test("prefers the wall clock over a monotonic counter", () => {
  const { header, data } = rows(HEARTSTUDIO);
  assert.equal(pickTimestampColumn(header, data), "observed_at_utc");
});

test("never selects a bare-integer column as the timestamp", () => {
  const { header, data } = rows(HEARTSTUDIO);
  assert.notEqual(pickTimestampColumn(header, data), "sensor_timestamp_seconds");
});

test("recognises an explicit UTC offset so it is not shifted twice", () => {
  assert.equal(hasExplicitOffset("2026-08-25T18:43:00.739Z"), true);
  assert.equal(hasExplicitOffset("2026-08-25T18:43:00+02:00"), true);
  assert.equal(hasExplicitOffset("2026-08-25 18:43:00"), false);
});

const DEXCOM = `Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Glucose Value (mg/dL)
1,2026-08-25T18:43:00,EGV,95
2,2026-08-25T18:48:00,EGV,92`;

test("still handles a stock Dexcom Clarity export", () => {
  const { header, data } = rows(DEXCOM);
  assert.equal(pickTimestampColumn(header, data), "Timestamp (YYYY-MM-DDThh:mm:ss)");
});

const LIBRE = `Device,Serial Number,Device Timestamp,Record Type,Historic Glucose mg/dL
FreeStyle,X1,25-08-2026 18:43,0,95
FreeStyle,X1,25-08-2026 18:58,0,92`;

test("still handles a FreeStyle Libre export", () => {
  const { header, data } = rows(LIBRE);
  assert.equal(pickTimestampColumn(header, data), "Device Timestamp");
});

test("reports no column rather than guessing when there are no dates", () => {
  const { header, data } = rows(`a,b\n1,2\n3,4`);
  assert.equal(pickTimestampColumn(header, data), undefined);
});

if (failures) {
  console.error(`\n${failures} test(s) failed`);
  process.exit(1);
}
console.log("\nall csv detection tests passed");
