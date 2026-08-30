#!/usr/bin/env python3
"""Stage the encoder and the probe heads for public distribution.

The website and the HTTP API serve these files to anyone who visits, so this is a
distribution of model weights and the licence of every training source has to survive it.

**The encoder** (`encoder.onnx`) is pretrained on Lane A only and is published CC-BY-NC-4.0.

**The heads** (`heads.json`) are 18 logistic classifiers fitted on frozen embeddings. Eight of
them are fitted on CGMacros, which the registry records as `CC-BY-NC-SA-4.0` with
`license_confidence: unverified` (open question Q4). Non-commercial is satisfied -- this is
unfunded research and the weights already forbid commercial use. Share-alike is not a
permission problem either; it is a labelling obligation: a work adapted from BY-NC-SA
material must itself be offered under BY-NC-SA.

So the heads bundle is published under **CC-BY-NC-SA-4.0**, one step more restrictive than
the encoder, and the two artefacts carry their own licences rather than being flattened into
one. That satisfies share-alike without withholding anything, and without letting the
share-alike term reach the encoder, which never saw CGMacros.

Heads fitted on Stanford ship as well. Stanford's licence is unresolved (registry value
`verify`, Q3), but it is Lane A: 171,140 of the corpus's 353,127 windows are Stanford and are
already inside the encoder. Withholding three classifiers while publishing the encoder that
pretrained on half that dataset would protect nothing. Q3 is a question about the encoder and
is tracked as one.

Recorded as decision D025.

    uv run python scripts/publish_web_assets.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
WEB_MODELS = REPO_ROOT / "web" / "public" / "models"

#: Sources that force share-alike on any bundle containing a classifier fitted on them.
SHAREALIKE_SOURCES = {"cgmacros", "glucofm_bench"}

#: Sources that may not be redistributed as a derivative at all (no-derivatives terms).
NO_DERIVATIVES_SOURCES = {"uchtt1dm"}


def main() -> int:
    WEB_MODELS.mkdir(parents=True, exist_ok=True)

    encoder = ARTIFACTS / "glucofm_encoder.onnx"
    if not encoder.exists():
        print(f"error: {encoder} missing; run `just export-onnx` first")
        return 2
    shutil.copy2(encoder, WEB_MODELS / "encoder.onnx")
    shutil.copy2(
        ARTIFACTS / "glucofm_encoder.onnx.meta.json", WEB_MODELS / "encoder.meta.json"
    )

    # Percentile breakpoints of each probe's score over a corpus sample. Without this the
    # site can only show which way a probe leans; with it, a day can be ranked against the
    # corpus, which is the statistic a ROC-AUC of 0.64-0.88 actually supports.
    # The streams build exposes the state/event decomposition, which is the architecture's
    # one load-bearing component (-0.0504 ROC when ablated) and the only part a visitor can
    # be shown rather than told about.
    streams = ARTIFACTS / "glucofm_encoder_streams.onnx"
    if streams.exists():
        shutil.copy2(streams, WEB_MODELS / "encoder_streams.onnx")
        shutil.copy2(
            ARTIFACTS / "glucofm_encoder_streams.onnx.meta.json",
            WEB_MODELS / "encoder_streams.meta.json",
        )

    reference = ARTIFACTS / "glucofm_reference.json"
    if reference.exists():
        shutil.copy2(reference, WEB_MODELS / "reference.json")
    else:
        print("  note: no reference distribution; run scripts/build_reference_distribution.py")

    bundle = json.loads((ARTIFACTS / "glucofm_heads.json").read_text())

    kept, withheld = {}, {}
    for key, head in bundle["heads"].items():
        if head["dataset"] in NO_DERIVATIVES_SOURCES:
            withheld[key] = (
                f"{head['dataset']}: no-derivatives licence — a fitted classifier may not "
                "be redistributed"
            )
        else:
            kept[key] = head

    sources = sorted({h["dataset"] for h in kept.values()})
    sharealike = sorted(SHAREALIKE_SOURCES.intersection(sources))
    bundle_license = "CC-BY-NC-SA-4.0" if sharealike else "CC-BY-NC-4.0"

    published = dict(bundle)
    published["heads"] = kept
    published["withheld"] = withheld
    published["license"] = bundle_license
    published["license_reason"] = (
        f"Share-alike inherited from {', '.join(sharealike)}; a classifier fitted on "
        "BY-NC-SA material must itself be offered under BY-NC-SA."
        if sharealike
        else "No share-alike source contributed to these heads."
    )
    published["license_note"] = (
        "This bundle is licensed separately from the encoder, which is CC-BY-NC-4.0 and was "
        "pretrained without any share-alike source. See scripts/publish_web_assets.py and D025."
    )
    published["fitted_on"] = sources
    (WEB_MODELS / "heads.json").write_text(json.dumps(published))

    n_signal = sum(1 for h in kept.values() if h["reliability"]["has_signal"])
    print(f"published {len(kept)} heads ({n_signal} with signal); withheld {len(withheld)}")
    print(f"heads bundle licence: {bundle_license}")
    print(f"  fitted on: {', '.join(sources)}")
    for key, reason in sorted(withheld.items()):
        print(f"  withheld  {key:<52} {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
