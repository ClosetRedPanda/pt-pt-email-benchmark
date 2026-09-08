"""Managed external model resources.

Some evaluator signals depend on model files that are too large to commit to
the repository. This module declares those resources in one place, verifies
them by SHA-256, and fetches them on demand.

Design constraints this follows:

* A resource is pinned to an exact upstream revision and an exact digest. A
  file that does not match its expected digest is rejected rather than used,
  so a corrupted or substituted download cannot silently change scores.
* Downloads never happen implicitly during scoring. `runner.py setup` fetches
  them explicitly, so a benchmark run never stalls on network I/O.
* The resolved path is reported to the manifest via `resource_paths`, binding
  every result to the exact model file that produced it.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent


class ResourceUnavailable(RuntimeError):
    """Raised when a required managed resource is absent or fails verification."""


@dataclass(frozen=True)
class ManagedResource:
    key: str
    relative_path: str
    url: str
    sha256: str
    size_bytes: int
    description: str
    license_note: str

    @property
    def path(self) -> Path:
        return BASE_DIR / self.relative_path


# `model_quantized.ftz` is the quantized EUPTVID fastText variety classifier.
# Pinned to an immutable commit rather than a branch: resolving `main` would
# let the upstream repository change the model under a fixed benchmark version.
EUPTVID = ManagedResource(
    key="euptvid",
    relative_path="models/model_quantized.ftz",
    url=(
        "https://huggingface.co/duarteocarmo/fasttext-euptvid/resolve/"
        "f58c96b242d63b2c48f98cfb00045e4cf8cdf6b0/model_quantized.ftz"
    ),
    sha256="00add97008d34b43803471daedb60d910de9b1eac15fc00814c336a4b23f0f6d",
    size_bytes=71170864,
    description="EUPTVID PT-PT/PT-BR fastText variety classifier (quantized)",
    license_note="MIT (duarteocarmo/fasttext-euptvid)",
)

MANAGED_RESOURCES: Dict[str, ManagedResource] = {EUPTVID.key: EUPTVID}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(resource: ManagedResource) -> Optional[str]:
    """Return an error string when the on-disk file is missing or wrong."""
    if not resource.path.is_file():
        return f"missing: {resource.relative_path}"
    actual = sha256_file(resource.path)
    if actual != resource.sha256:
        return (
            f"checksum mismatch for {resource.relative_path}: "
            f"expected {resource.sha256}, found {actual}"
        )
    return None


def is_available(resource: ManagedResource) -> bool:
    return verify(resource) is None


def require(resource: ManagedResource) -> Path:
    """Return a verified resource path or raise with actionable instructions."""
    problem = verify(resource)
    if problem is None:
        return resource.path
    raise ResourceUnavailable(
        f"{problem}\n"
        f"  {resource.description}\n"
        f"  Fetch it with:  python runner.py setup\n"
        f"  License: {resource.license_note}"
    )


def download(resource: ManagedResource, *, force: bool = False) -> Path:
    """Download and verify a managed resource.

    The file is written to a temporary path and only moved into place after its
    digest matches, so an interrupted download cannot leave a half-written file
    that later looks present.
    """
    if not force and is_available(resource):
        return resource.path

    resource.path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] downloading {resource.key} "
        f"({resource.size_bytes / 1e6:.0f} MB) -> {resource.relative_path}",
        file=sys.stderr,
    )
    tmp_handle = tempfile.NamedTemporaryFile(
        delete=False, dir=str(resource.path.parent), suffix=".part"
    )
    tmp_path = Path(tmp_handle.name)
    tmp_handle.close()
    try:
        request = urllib.request.Request(
            resource.url, headers={"User-Agent": "pt-pt-email-benchmark"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with tmp_path.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise ResourceUnavailable(
            f"failed to download {resource.key} from {resource.url}: {exc}"
        ) from exc

    actual = sha256_file(tmp_path)
    if actual != resource.sha256:
        tmp_path.unlink(missing_ok=True)
        raise ResourceUnavailable(
            f"downloaded {resource.key} failed verification: "
            f"expected {resource.sha256}, got {actual}"
        )
    tmp_path.replace(resource.path)
    print(f"[setup] verified {resource.relative_path}", file=sys.stderr)
    return resource.path


def download_all(*, force: bool = False) -> Dict[str, str]:
    status: Dict[str, str] = {}
    for key, resource in MANAGED_RESOURCES.items():
        try:
            download(resource, force=force)
            status[key] = "ok"
        except ResourceUnavailable as exc:
            status[key] = f"FAILED: {exc}"
    return status
