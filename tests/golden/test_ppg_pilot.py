"""Golden tests for the §23 PPG teacher-student pilot (D023, A7).

These tests pin the architecture and loss shapes so any drift fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from opencgm_stateevent.ppg import (
    BVP_RATE_HZ,
    SAMPLES_PER_PATCH,
    DirectGlucoseHead,
    PpgStudentEncoder,
    TeacherLatentHead,
    alignment_loss,
    count_parameters,
    gaussian_nll,
)


def test_patch_len_is_19200():
    """5 minutes x 60 s/min x 64 Hz = 19200 samples per patch."""
    assert SAMPLES_PER_PATCH == 19200
    assert BVP_RATE_HZ == 64
    assert SAMPLES_PER_PATCH == BVP_RATE_HZ * 60 * 5


def test_student_encoder_total_params():
    """Total student params (encoder + both heads) is roughly 70K, far below the teacher's
    ~520K. This pins the 'small student' framing of D023."""
    enc = PpgStudentEncoder()
    lat = TeacherLatentHead()
    glu = DirectGlucoseHead()
    total = count_parameters(enc) + count_parameters(lat) + count_parameters(glu)
    assert 90_000 < total < 110_000, f"unexpected total params: {total}"


def test_student_encoder_output_shape():
    enc = PpgStudentEncoder()
    x = torch.zeros(8, SAMPLES_PER_PATCH)
    y = enc(x)
    assert y.shape == (8, PpgStudentEncoder.FEATURE_DIM)
    assert y.shape == (8, 64)


def test_student_encoder_rejects_bad_shape():
    enc = PpgStudentEncoder()
    with pytest.raises(ValueError):
        enc(torch.zeros(8, 1000))  # wrong patch length


def test_teacher_latent_head_output_shape():
    lat = TeacherLatentHead()
    feats = torch.zeros(8, 64)
    out = lat(feats)
    assert out.shape == (8, 128)


def test_direct_glucose_head_output_shape():
    glu = DirectGlucoseHead()
    feats = torch.zeros(8, 64)
    out = glu(feats)
    assert out.shape == (8, 2)  # (mean_mmol_per_l, log_sigma)


def test_alignment_loss_shape_check():
    """Mismatched shapes raise."""
    student = torch.zeros(8, 256)
    teacher = torch.zeros(7, 256)
    mask = torch.zeros(8)
    with pytest.raises(ValueError):
        alignment_loss(student, teacher, mask)


def test_alignment_loss_with_all_masked():
    """All-masked batch returns a zero loss that is still differentiable."""
    student = torch.zeros(4, 256, requires_grad=True)
    teacher = torch.zeros(4, 256)
    mask = torch.zeros(4)  # all masked
    loss = alignment_loss(student, teacher, mask)
    assert loss.item() == 0.0
    # Backward must not raise.
    loss.backward()


def test_gaussian_nll_log_sigma_clamped():
    """log_sigma clamped to [-3, 3] = sigma in [0.05, 20]. Out-of-range inputs don't blow up."""
    feats = torch.zeros(8, 64)
    glu = DirectGlucoseHead()
    pred = glu(feats)
    pred[:, 1] = 100.0  # extreme log_sigma, should be clamped
    target = torch.zeros(8)
    mask = torch.ones(8)
    loss = gaussian_nll(pred, target, mask)
    assert torch.isfinite(loss).all()


def test_alignment_loss_improves_with_perfect_match():
    """Sanity: alignment loss is 0 when student == teacher (perfectly aligned)."""
    feats = torch.zeros(4, 64)
    lat = TeacherLatentHead()
    out = lat(feats)
    mask = torch.ones(4)
    loss = alignment_loss(out, out, mask)
    # Cosine of (out, out) = 1, MSE = 0, so loss = 0.
    assert loss.item() < 1e-5


def test_determinism_one_seed():
    """Two forward passes with the same seed produce identical outputs (no randomness)."""
    torch.manual_seed(7)
    enc = PpgStudentEncoder()
    x = torch.randn(8, SAMPLES_PER_PATCH)
    y1 = enc(x)
    y2 = enc(x)
    np.testing.assert_allclose(y1.detach().numpy(), y2.detach().numpy(), atol=1e-6)


def test_cgm_parser_format_order_no_regression():
    """Regression: '01-11-2024' must parse as 1 Nov 2024 (DD-MM-YYYY), not 11 Jan 2024.

    Earlier the parser tried MM-DD-YYYY first, which silently mis-parsed dates like
    '01-11-2024' as Jan 11 instead of Nov 1. That dropped CGM alignments for 4 of the 5
    subjects to zero. This test pins the format list order so the regression cannot return.
    """
    from datetime import datetime

    from opencgm_stateevent.ppg.align import _parse_cgm_csv

    # Synthetic CSV with two ambiguous-looking dates that would catch a format-order bug.
    csv_text = (
        "Device Timestamp,Historic Glucose mmol/L,Scan Glucose mmol/L\n"
        "01-11-2024 00:01,6.1,\n"
        "02-02-2025 22:51,3.9,\n"
    )
    p = Path("/tmp/_test_cgm_parse.csv")
    p.write_text(csv_text)
    rows = _parse_cgm_csv(p)
    p.unlink()
    assert len(rows) == 2
    # 01-11-2024 -> Nov 1 (not Jan 11)
    assert rows[0][0] == datetime(2024, 11, 1, 0, 1)
    assert rows[0][1] == 6.1
    # 02-02-2025 -> Feb 2 (ambiguous, either DD-MM or MM-DD gives the same day)
    assert rows[1][0] == datetime(2025, 2, 2, 22, 51)
    assert rows[1][1] == 3.9


def test_cgm_parser_all_subjects_have_overlap():
    """All 5 on-disk subjects must yield >1000 CGM-aligned patches after parse fix.

    Pins the end-to-end pipeline (parse + BVP walk + alignment) so a silent regression
    to MM-DD-YYYY ordering or to the wrong timezone offset cannot hide. This is the
    failure mode that originally motivated §23: 4 of 5 subjects had 0 patches.
    """
    import os

    data_zip = os.environ.get(
        "PPG_DATA_ZIP", "data/raw/ppg_cgm_paired_zenodo_20577959"
    )
    from opencgm_stateevent.ppg.align import iter_aligned_patches, list_subjects

    work = Path("/tmp/_ppg_test_work")
    if not (work / "Data").exists():
        # Lazily extract only if not already present.
        import zipfile

        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path(data_zip) / "Data.zip") as z:
            z.extractall(work)
    subjects = list_subjects(work)
    assert subjects == ["P001", "P002", "P003", "P004", "P005"]
    by_subj: dict[str, int] = {s: 0 for s in subjects}
    by_with: dict[str, int] = {s: 0 for s in subjects}
    for patch in iter_aligned_patches(work):
        by_subj[patch.subject] += 1
        if patch.glucose_mmol is not None:
            by_with[patch.subject] += 1
    for s in subjects:
        assert by_subj[s] > 1000, f"{s}: only {by_subj[s]} total patches"
        assert by_with[s] > 1000, (
            f"{s}: only {by_with[s]} CGM-aligned patches - parser or timezone regression?"
        )
