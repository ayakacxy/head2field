"""T1 preprocessing profile constants without SciPy or SimNIBS dependencies."""

from __future__ import annotations

HCP_ACPC_PROFILE = "hcp_acpc"
NON_HCP_RAW_PROFILE = "non_hcp_raw"
T1_PREPROCESSING_PROFILES = (HCP_ACPC_PROFILE, NON_HCP_RAW_PROFILE)

__all__ = [
    "HCP_ACPC_PROFILE",
    "NON_HCP_RAW_PROFILE",
    "T1_PREPROCESSING_PROFILES",
]
