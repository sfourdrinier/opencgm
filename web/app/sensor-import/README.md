<!-- web/app/sensor-import/README.md -->

# Browser sensor import

This is the operator and support note for the direct browser importer at
<https://opencgm.vercel.app/sensor-import>. The route is intentionally reachable
by direct URL while hardware proof is in progress; it is not linked from the
site navigation, home page, try page, or sitemap.

## What a user needs

The primary UI requirement is:

> Requires Google Chrome, Bluetooth, and a Dexcom G7 sensor.

For a new sensor, enter the four-digit pairing code and use the Connect button
to start the operation. The first browser Bluetooth operation is the chooser,
so it must stay directly under that click. The browser's own capability and
permission checks are authoritative. In production, use the HTTPS host above.
`http://localhost` is supported for development only because localhost is a
browser secure-context exception; an arbitrary HTTP deployment is not
supported.

## What happens locally

The browser talks directly to the selected sensor through Web Bluetooth. It
loads the first-party WASM artifact, normalizes the readings, and runs the same
analysis boundary used by the site in the browser. Sensor readings,
credentials, pairing material, and downloaded exports are not posted to an API
or relay. The browser still downloads static application/model assets from the
site; that is asset loading, not an upload of sensor data.

There is no background monitor. The connection exists only for the active
operation and is stopped when the user cancels, disconnects, leaves the page, or
the link ends. A same-origin compromise remains in scope: an injected or
malicious script already executing as this origin could inspect JavaScript
memory, WebAssembly memory, DOM state, or local storage. CSP and artifact
integrity checks reduce accidental and supply-chain exposure, but cannot repair
an already-compromised origin.

## Credentials and local persistence

The page can accept a pairing code, an exported remembered key, or an opaque
certificate bundle when a session requires one. These values stay in the tab
and are passed only to the local engine. Diagnostic key and certificate inputs
are available only with the exact `?debug=yesplease` query.

Remembering a key is a separate opt-in checkbox and is off by default. When
IndexedDB is available, the credential vault encrypts the key with WebCrypto
AES-GCM and stores the non-extractable key plus ciphertext locally, scoped to
the browser-authorized sensor identity. While this page remains open, it can
automatically reconnect when exactly one authorized sensor has a remembered
credential; it does not guess when none or multiple sensors are eligible.
Choose another sensor to open the chooser and enter the code again. If
persistent storage is unavailable, the vault falls back to a session-only
in-memory choice. “Forget local key” removes the stored record.
Browser memory, browser backups, extensions, developer tools, and malicious
same-origin code are outside this protection; do not treat the vault as a
device security boundary.

## Support and limitations

The sensor becomes available briefly about every five minutes. Keep this page
open; discovery can take several minutes. A sleeping or unavailable remembered
sensor remains a waiting state. Stop the competing session, bring the sensor
nearby, and use Choose another sensor if needed. A cancelled
chooser, denied permission, missing credential, or lost link can be retried
locally; the page does not bypass the browser's Bluetooth permission model.

Chrome cannot complete a new operating-system Bluetooth bond through this page.
The first pairing can still reach the sensor protocol's pairing step, but a
browser-authorized sensor with prepared pairing information or an existing host
bond may be required at that boundary.

The importer returns only the history the sensor makes available. History may
have expired or the link may end before the requested range is delivered. The
result then remains usable but is marked `partial` and carries warnings. A
partial result does not prove that an unavailable interval had no readings.
Analysis also needs enough reliable readings to form a scoreable day, so a
sparse import can succeed while analysis remains unavailable.

Completed imports can be downloaded as CSV or NDJSON from the page. Downloads
are created locally by the browser and include the reported completeness and
warnings in the NDJSON metadata. Review an export before sharing it: it contains
health data and sensor identifiers.

This is an experimental research instrument, not a medical device or medical
advice. It is not validated for diagnosis, treatment, dosing, alerts, or
continuous monitoring. Keep using an approved receiver/app for safety-critical
decisions, and never assume that a successful browser import is complete or
current.

## Artifact and licensing

The public deployment contains one audited production module,
`/sensor/sensor-engine.abi1.wasm`, and a matching manifest. The loader verifies
the manifest digest before use. The module is deliberately opaque and stripped;
WASM is not a secrecy mechanism. Release provenance and the complete dependency
notice are recorded in the repository [`NOTICE`](../../../NOTICE). The public
checkout does not contain private engine source, protocol internals, wire
traces, private vectors, credentials, or source maps.

The release artifact's behavior is MIT-derived from the public LoopKit reference
projects listed in `NOTICE`; the engine's permissively licensed Rust dependency
notices are listed there as well. This project is independent of those projects
and of the sensor manufacturer.
