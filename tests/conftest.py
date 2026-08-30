"""Pytest configuration: expose the `scripts/` package to `import scripts.<name>`.

The export scripts live under `scripts/` and are invoked as `python scripts/<name>.py`, not as
`python -m scripts.<name>`. The parity tests want to reuse the `EncoderMeanEmbed` wrapper and
its decomposed transformer from `scripts/export_encoder_onnx.py`, so we add the *parent* of
`scripts/` to `sys.path` here once, at session start.

Why the parent, not `scripts/` itself? `PathFinder.find_spec('scripts')` walks each
`sys.path` entry looking for either `<entry>/scripts.py` (file) or `<entry>/scripts/__init__.py`
(package). If `<entry>` IS the scripts directory, neither of those matches and the import
fails. Inserting the repo root lets PathFinder find `<repo>/scripts/__init__.py` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Insert at front so `import scripts.export_encoder_onnx` resolves before any installed
# package of the same name. Idempotent: pytest re-runs conftest on collection, so check first.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
