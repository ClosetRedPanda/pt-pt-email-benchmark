"""Dependency-free helpers shared across the benchmark core.

Small, single-definition utilities that several modules previously duplicated:
file hashing, percentile interpolation, and a None-preserving mean.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed so large resources are not fully loaded."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    """Arithmetic mean of finite values, or None when none are available."""
    vals = [float(v) for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    """Linear-interpolated percentile; None for an empty sample."""
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    frac = k - lo
    return vals[lo] + (vals[hi] - vals[lo]) * frac
