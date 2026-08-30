"""Command line entry point. Blueprint §25.

Every command writes a machine-readable run record. `data status` is the project's
source of truth: it reads the filesystem, never a hand-maintained checklist.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import EvidenceError, load_config
from .provenance import RunRecord, sha256_file
from .sources import LANE_LABEL, Lane, scan, strict_corpus_summary

app = typer.Typer(no_args_is_help=True, add_completion=False, help="OpenCGM-StateEvent")
data_app = typer.Typer(no_args_is_help=True, help="Source acquisition and canonicalization")
app.add_typer(data_app, name="data")

console = Console()

# Hoisted so the Option objects are singletons rather than per-call constructions (ruff B008).
REGISTRY_OPT = typer.Option(Path("manifests/sources/registry.yaml"), help="Source registry")
LANE_OPT = typer.Option(None, help="Filter to one lane: A B C D E")
DATASET_OPT = typer.Option(None, help="Only this dataset_id")


@app.command()
def version() -> None:
    """Print version."""
    console.print(f"opencgm-stateevent {__version__}")


@data_app.command("status")
def data_status(
    registry: Path = REGISTRY_OPT,
    lane: str | None = LANE_OPT,
) -> None:
    """What is on disk, under which rights, in which lane."""
    rec = RunRecord(command="data status", inputs={"registry": str(registry)})
    statuses = scan(registry)
    if lane:
        statuses = [s for s in statuses if s.lane is Lane(lane.upper())]

    for lane_key in [Lane.A, Lane.B, Lane.C, Lane.D, Lane.E]:
        rows = [s for s in statuses if s.lane is lane_key]
        if not rows:
            continue
        table = Table(
            title=f"Lane {lane_key} — {LANE_LABEL[lane_key]}",
            title_justify="left",
            header_style="bold",
        )
        table.add_column("dataset")
        table.add_column("on disk")
        table.add_column("files", justify="right")
        table.add_column("size", justify="right")
        table.add_column("cgm", justify="right")
        table.add_column("license")
        table.add_column("weights")
        for s in rows:
            ok = "[green]yes[/]" if s.present else "[red]MISSING[/]"
            weights = (
                "[green]ok[/]" if s.distributable else f"[red]{s.weight_release.split('_')[-1]}[/]"
            )
            size = f"{s.gb:.2f} GB" if s.bytes >= 1e9 else f"{s.bytes / 1e6:.1f} MB"
            table.add_row(
                s.dataset_id, ok, str(s.n_files), size,
                str(s.cgm_matches) if s.cgm_matches else "-", s.license, weights,
            )
        console.print(table)
        console.print()

    summary = strict_corpus_summary(scan(registry))
    console.print(
        f"[bold]Strict corpus (Lane A):[/] "
        f"{summary['sources_present']}/{summary['sources_total']} sources on disk · "
        f"target {summary['paper_target_records']} records / "
        f"{summary['paper_target_hours']:,} h "
        f"(~30.9% of the paper's pretraining hours; Wear-CGM is non-public)"
    )
    blocked = [s.dataset_id for s in statuses if not s.distributable]
    if blocked:
        console.print(
            f"[yellow]Never in a distributed checkpoint:[/] {', '.join(blocked)} "
            "— see bundle/BLUEPRINT_AMENDMENTS.md A4"
        )
    rec.finish(sources=len(statuses), **summary).write()


@data_app.command("verify")
def data_verify(
    registry: Path = REGISTRY_OPT,
    dataset: str | None = DATASET_OPT,
) -> None:
    """Hash the acquired CGM files so drift is detectable later."""
    rec = RunRecord(command="data verify")
    statuses = scan(registry)
    if dataset:
        statuses = [s for s in statuses if s.dataset_id == dataset]
    total = 0
    for s in statuses:
        if not s.present:
            console.print(f"[red]MISSING[/] {s.dataset_id}")
            continue
        reg = next(
            e for e in __import__("yaml").safe_load(registry.read_text())["sources"]
            if e["dataset_id"] == s.dataset_id
        )
        files = sorted(s.path.glob(reg.get("cgm_glob", "")))[:200]
        for f in files:
            console.print(f"  {sha256_file(f)[:16]}  {f.relative_to(s.path)}")
            total += 1
    console.print(f"[bold]{total}[/] files hashed")
    rec.finish(files_hashed=total).write()


@app.command("config-check")
def config_check(path: Path) -> None:
    """Validate a config's evidence tags and list its inferred choices."""
    try:
        cfg = load_config(path)
    except EvidenceError as e:
        console.print(f"[red]FAIL[/] {e}")
        raise typer.Exit(1) from e
    console.print(f"[green]OK[/] {path}  hash={cfg.config_hash[:16]}")
    inferred = cfg.inferred_choices()
    if inferred:
        console.print(
            f"\n[yellow]{len(inferred)} INFERRED_RECONSTRUCTION choices[/] "
            "— each must appear in DECISIONS.md before agent fan-out:"
        )
        for path_str in inferred:
            console.print(f"  · {path_str}")


if __name__ == "__main__":
    app()
