import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { ExampleAnalysis } from "./example";

// One loader for the worked example, used by both /example and the front page.
//
// Both pages are statically prerendered, so this runs at build time and the JSON never has to
// be fetched by a visitor. Keeping it in one place means the front page cannot drift onto a
// stale copy of the data the example page is showing.

let cache: ExampleAnalysis | null = null;

export function loadExample(): ExampleAnalysis {
  if (!cache) {
    cache = JSON.parse(
      readFileSync(join(process.cwd(), "public", "data", "example-analysis.json"), "utf8"),
    ) as ExampleAnalysis;
  }
  return cache;
}
