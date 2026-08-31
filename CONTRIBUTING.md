# Contributing

Contributions are welcome. OpenCGM-StateEvent is published under Apache-2.0; any
contribution is taken under the same license unless you explicitly mark it otherwise.

## Before you open a PR

1. **`just gate`** must pass locally. This runs ruff + pytest + `just status`. Status
   reads the filesystem; if a number in the docs disagrees with `just status`, the status
   output is right.
2. **No dataset blobs.** Never commit data from `data/raw/`, fitted heads from
   `artifacts/heads.pkl`, or checkpoints from `runs*/`. These are gitignored for a
   reason — see `DECISIONS.md` D001.
3. **No Lane E in distributed artefacts.** If your change touches training, evaluation, or
   the released encoder, check that no Lane E source (cgmacros, uchtt1dm, glucofm_bench)
   has crept into a shipped artefact. `tests/unit/test_source_rights.py` enforces this.
4. **Never interpolate CGM.** Missing data is masked, never filled. Tests in
   `tests/golden/test_no_leakage.py` and `tests/golden/test_grid.py` enforce this.
5. **Tag every consequential choice.** When you fork a design decision into a new
   approach, write it into `DECISIONS.md` with the rationale and the
   `PAPER_EXACT | SOURCE_VERIFIED | INFERRED_RECONSTRUCTION | PROPOSED_EXTENSION` tag
   *before* you push the code.

## Style

- **Python:** ruff-enforced, line length 100. `from __future__ import annotations` is
  preferred for `src/` but not required.
- **Type hints:** required in new code in `src/`.
- **Tests:** every new module in `src/` ships with at least one unit test in `tests/unit/`.
  Numerical-parity invariants ship as `@pytest.mark.golden` tests in `tests/golden/`.

## Release process

1. Record any consequential choice in `DECISIONS.md`, with the alternatives.
2. Bump `version` in `pyproject.toml` if the change is user-visible.
3. Run `just gate` and `just status`. Commit.
4. Tag the release: `git tag -a vMAJOR.MINOR.PATCH -m "..."`.
5. Push. The CI workflow in `.github/workflows/ci.yml` runs lint + tests on every push.

## Reporting a security issue

Open a GitHub Security Advisory on the `sfourdrinier/opencgm-stateevent` repository.
Do not open a public issue for anything that touches participant data.
