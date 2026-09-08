# Managed model resources

This directory is **gitignored** (`models/*`): the model binaries are large and
are intentionally not committed. The provenance of anything fetched into this
directory is documented below so results stay reproducible.

The only managed resource is:

| Resource | File | Source | Pinned revision | SHA-256 (full) | Size (bytes) | License |
| --- | --- | --- | --- | --- | --- | --- |
| EUPTVID | `models/model_quantized.ftz` | `huggingface.co/duarteocarmo/fasttext-euptvid` | `f58c96b242d63b2c48f98cfb00045e4cf8cdf6b0` | `00add97008d34b43803471daedb60d910de9b1eac15fc00814c336a4b23f0f6d` | 71,170,864 | MIT |

Authoritative metadata lives in `core/resources.py` (a `ManagedResource`
definition). If the two differ, treat `core/resources.py` as the source of truth.

## Fetching and verifying

```bash
python runner.py setup
```

`runner.py setup` downloads the file pinned to the **exact immutable revision**
above (a commit, not a branch, so upstream cannot move the model under a fixed
benchmark version) and verifies its SHA-256 before use. Re-run it any time; it
is a no-op when the file is already present and valid. The installed digest is
recorded in every run manifest.

If the file is absent or fails verification, the EUPTVID signal is reported as
**unavailable** rather than a fabricated score (see `docs/VALIDATION.md`).
