"""Public surface of the Lane-D PPG pilot (D023, A7)."""

from .align import (
    BVP_RATE_HZ,
    SAMPLES_PER_PATCH,
    BvpCgmPatch,
    iter_aligned_patches,
    list_subjects,
)
from .conditional_teacher import (
    CENTER_PATCH,
    CENTER_POSITION,
    CgmContext,
    build_cgm_context,
    encode_center_token,
    precompute_teacher_targets,
)
from .encoder import PpgStudentEncoder, count_parameters
from .heads import (
    DirectGlucoseHead,
    TeacherLatentHead,
    alignment_loss,
    gaussian_nll,
)

__all__ = [
    "BVP_RATE_HZ",
    "CENTER_PATCH",
    "CENTER_POSITION",
    "SAMPLES_PER_PATCH",
    "BvpCgmPatch",
    "CgmContext",
    "DirectGlucoseHead",
    "PpgStudentEncoder",
    "TeacherLatentHead",
    "alignment_loss",
    "build_cgm_context",
    "count_parameters",
    "encode_center_token",
    "gaussian_nll",
    "iter_aligned_patches",
    "list_subjects",
    "precompute_teacher_targets",
]
