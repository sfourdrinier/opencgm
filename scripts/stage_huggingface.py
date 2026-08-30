#!/usr/bin/env python3
"""Assemble the Hugging Face model repository, without publishing it.

Publishing is deliberately not automated. Uploading weights is irreversible in practice --
people fetch them, mirror them, and cite them -- and it is the author's decision, not a
script's. This stages a directory and prints the exact command; a human runs it.

    uv run python scripts/stage_huggingface.py
    hf auth login                      # once
    hf upload sfourdrinier/opencgm-stateevent dist/hf . --repo-type model

What goes in:
  * `glucofm_encoder.onnx` and its provenance sidecar
  * `glucofm_heads.json`, exactly as the website serves it
  * `README.md` -- the model card, which already carries the HF YAML front matter
  * `LICENSE-WEIGHTS` (encoder, CC-BY-NC-4.0) and `LICENSE-HEADS` (heads, CC-BY-NC-SA-4.0)
  * the repository NOTICE

What stays out, and why:
  * PyTorch `.pt` checkpoints. They are 11 MB each and carry optimiser state; the ONNX is
    the artefact people can actually run. Publishing the training checkpoints is a separate
    decision with its own size and licence consequences.
  * Nothing fitted on UCHTT1DM, whose no-derivatives term admits no labelling remedy. There
    is no such head today; publish_web_assets.py enforces it regardless.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = REPO_ROOT / "dist" / "hf"
HF_REPO = "sfourdrinier/opencgm-stateevent"


def main() -> int:
    published_heads = REPO_ROOT / "web" / "public" / "models" / "heads.json"
    if not published_heads.exists():
        print("error: run `just publish-web-assets` first (it applies the rights filter)")
        return 2

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    files = [
        (REPO_ROOT / "artifacts" / "glucofm_encoder.onnx", "glucofm_encoder.onnx"),
        (
            REPO_ROOT / "artifacts" / "glucofm_encoder.onnx.meta.json",
            "glucofm_encoder.onnx.meta.json",
        ),
        (published_heads, "glucofm_heads.json"),
        (REPO_ROOT / "model_cards" / "glucofm_encoder.md", "README.md"),
        (REPO_ROOT / "LICENSE-WEIGHTS", "LICENSE-WEIGHTS"),
        (REPO_ROOT / "LICENSE-HEADS", "LICENSE-HEADS"),
        (REPO_ROOT / "NOTICE", "NOTICE"),
        (REPO_ROOT / "CITATION.cff", "CITATION.cff"),
    ]

    total = 0
    for src, name in files:
        if not src.exists():
            print(f"error: missing {src}")
            return 2
        shutil.copy2(src, STAGE / name)
        total += src.stat().st_size
        print(f"  staged {name:<34} {src.stat().st_size / 1024:>8.1f} KB")

    card = (STAGE / "README.md").read_text()
    if not card.startswith("---"):
        print("error: model card has no YAML front matter; HF will not read the licence tag")
        return 2

    print(f"\n{len(files)} files, {total / 1e6:.2f} MB staged in {STAGE}")
    print("\nReview the card, then publish with:\n")
    print("  hf auth login")
    print(f"  hf upload {HF_REPO} {STAGE.relative_to(REPO_ROOT)} . --repo-type model\n")

    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if rev.returncode == 0:
        print(f"Staged from commit {rev.stdout.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
