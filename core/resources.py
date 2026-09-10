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

import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from core._util import sha256_file

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
# Revisions and digests are immutable benchmark inputs; changing any of them
# requires a benchmark/evaluator version bump and makes manifests intentionally
# incompatible.
EUPTVID = ManagedResource(
    key="euptvid",
    relative_path="models/model_quantized.ftz",
    url=(
        "https://huggingface.co/duarteocarmo/fasttext-euptvid/resolve/"
        "f58c96b242d63b2c48f98cfb00045e4cf8cdf6b0/model_quantized.ftz"
    ),
    sha256="00add97008d34b43803471daedb60d910de9b1eac15fc00814c336a4b23f0f6d",
    size_bytes=71_170_864,
    description="EUPTVID PT-PT/PT-BR fastText variety classifier (quantized)",
    license_note="MIT; upstream: https://huggingface.co/duarteocarmo/fasttext-euptvid",
)

# LibreOffice dictionary files are deliberately not redistributed by this
# Apache-2.0 repository. Setup fetches the exact bytes that were previously
# bundled, from one immutable LibreOffice commit, and verifies each SHA-256
# before an evaluator may load it.
_LIBREOFFICE_DICTIONARIES_REVISION = "1e848fbddd7fd8e03fb696ecc03ee1068fab141c"
_LIBREOFFICE_RAW = (
    "https://raw.githubusercontent.com/LibreOffice/dictionaries/"
    f"{_LIBREOFFICE_DICTIONARIES_REVISION}"
)
_HUNSPELL_LICENSE = (
    "GPL-2.0-or-later OR LGPL-2.1-or-later OR MPL-1.1; "
    "upstream: https://github.com/LibreOffice/dictionaries"
)

HUNSPELL_PT_PT_AFF = ManagedResource(
    key="hunspell_pt_pt_aff",
    relative_path="docs/pt_PT.aff",
    url=f"{_LIBREOFFICE_RAW}/pt_PT/pt_PT.aff",
    sha256="975a209fcc892cb382fa5f34a28c391a39668661ce373ae071287809c5fcae24",
    size_bytes=95_089,
    description="LibreOffice Hunspell affix rules for European Portuguese",
    license_note=_HUNSPELL_LICENSE,
)
HUNSPELL_PT_PT_DIC = ManagedResource(
    key="hunspell_pt_pt_dic",
    relative_path="docs/pt_PT.dic",
    url=f"{_LIBREOFFICE_RAW}/pt_PT/pt_PT.dic",
    sha256="e29ba2d7aa8a2ad43e9cb46ac6473064b661545c87002aea90e18899d98d3cc9",
    size_bytes=1_485_977,
    description="LibreOffice Hunspell word list for European Portuguese",
    license_note=_HUNSPELL_LICENSE,
)
HUNSPELL_PT_BR_AFF = ManagedResource(
    key="hunspell_pt_br_aff",
    relative_path="docs/pt_BR.aff",
    url=f"{_LIBREOFFICE_RAW}/pt_BR/pt_BR.aff",
    sha256="21d8ad2a769a60e17e2b5ea4ef11d4d593a58b9e2a82d642ef82d6a4c5523865",
    size_bytes=979_792,
    description="LibreOffice Hunspell affix rules for Brazilian Portuguese",
    license_note=_HUNSPELL_LICENSE,
)
HUNSPELL_PT_BR_DIC = ManagedResource(
    key="hunspell_pt_br_dic",
    relative_path="docs/pt_BR.dic",
    url=f"{_LIBREOFFICE_RAW}/pt_BR/pt_BR.dic",
    sha256="a38bfb26b68ece2834e79fe83e48d5792652970ace12db89d1b9674bf9933183",
    size_bytes=4_477_695,
    description="LibreOffice Hunspell word list for Brazilian Portuguese",
    license_note=_HUNSPELL_LICENSE,
)

HUNSPELL_RESOURCES = (
    HUNSPELL_PT_PT_AFF,
    HUNSPELL_PT_PT_DIC,
    HUNSPELL_PT_BR_AFF,
    HUNSPELL_PT_BR_DIC,
)
MANAGED_RESOURCES: Dict[str, ManagedResource] = {
    resource.key: resource for resource in (EUPTVID, *HUNSPELL_RESOURCES)
}


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
