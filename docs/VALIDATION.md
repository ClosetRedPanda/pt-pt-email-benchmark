# Validation status

This lean package deliberately makes fewer claims than the larger engineering versions.

## What is frozen
- A 20-row structured-analysis reference set with explicit ground truth.
- A fixed PT-PT generation prompt set and explicit executable constraints.
- A local deterministic WQ engine with a frozen calibration model.
- Separate PT-PT classifier/compliance, PT-BR leakage, WF, and WQ signals.

## What is validated automatically
- JSON-schema validity of the reference records.
- Constraint-file integrity and regex loading.
- Deterministic dialect/WF/WQ smoke checks.
- Reproducibility of the scoring code from plain JSONL artifacts.
- New result artifacts have strict row validation and a hash-bound sidecar manifest.
- `compare.py` rejects missing manifests by default and rejects incompatible
	manifest metadata before comparison.

## What is *not* claimed
The included WQ human reference contains 50 verified rows, but it does not establish multi-rater inter-rater reliability. The benchmark therefore does **not** call the formal human WQ gate passed.

EUPTVID is a managed external model asset: it is not committed to the repository, but `python runner.py setup` fetches it from a pinned upstream revision and verifies its SHA-256 before use, and its digest is recorded in every run manifest. A model that is absent or fails verification yields an unavailable signal rather than a fabricated one. Hunspell dictionaries are loaded from `docs/pt_PT.*` and `docs/pt_BR.*`; when unavailable, those signals remain unavailable rather than being replaced with fabricated scores.

Existing `v2` result files are legacy artifacts. They can be inspected only with
`--allow-legacy`; `--rescore` is explicitly labeled exploratory and does not yet
write a new immutable artifact. To persist one explicitly, use
`tools/rescore_artifact.py`, which writes a new JSONL file and a provenance
manifest containing the original artifact hash and evaluator source hashes.

## Philosophy
Raw outputs are retained. Automatic evaluators are instruments, not ground truth. There is no overall quality score in the primary result.
