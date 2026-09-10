# Managed model resources

This directory is **gitignored** (`models/*`): the model binaries are large and
are intentionally not committed. The only managed resource is the EUPTVID
fastText variety classifier (`models/model_quantized.ftz`, MIT,
`huggingface.co/duarteocarmo/fasttext-euptvid`).

Fetch and verify it with:

```bash
python runner.py setup
```

The exact pinned revision, SHA-256, and byte size live in `core/resources.py`
(the single authoritative `ManagedResource` definition). `runner.py setup`
downloads the pinned revision and verifies its SHA-256 before use; if the file
is absent or fails verification, the EUPTVID signal is reported as unavailable
rather than a fabricated score (see `docs/VALIDATION.md`).
