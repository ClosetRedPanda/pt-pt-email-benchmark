# Contributing

Thanks for considering a contribution to the PT-PT Email LLM Benchmark.

## Scope of contributions

This project is a **benchmark**, not an application. Changes that make the
scientific measurement *stronger, more reproducible, or more auditable* are
welcome. Changes that only add convenience tooling around the benchmark are
generally out of scope and belong in a companion project.

Concretely, useful contributions include:

- Expanding or improving the fixed reference and task datasets (see below).
- Fixing deterministic evaluators (dialect, WF, writing quality).
- Adding independent validation or calibration evidence.
- Improving reproducibility (CI, pinned resources, manifests).

## Dataset contributions

The reference set is deliberately small and frozen so that scores stay
comparable over time. If you want to add rows:

1. Add a row to `data/analysis_reference.jsonl` or the generation task files
   with explicit ground truth.
2. Make sure `python runner.py validate` passes (it asserts row counts and
   schema validity — adjust the count assertion deliberately).
3. Update `data/PROVENANCE.json` and `docs/DATA_PROVENANCE.md` with the row
   source, synthetic/LLM-assistance details, licence, prompts, and review method.

Ground truth must be *independently verifiable*, not authored by the person
writing the evaluator that is tested against it.

## Determinism and honesty

- Deterministic evaluators must never **fabricate** a score when evidence is
  missing. Missing resources must yield `None`/unavailable, not a guess.
- Dialect issues and writing-quality issues are scored separately. Do not
  merge or double-penalise them.
- Do not add LLM-judge scoring to the deterministic path.

## Tests

- Unit tests live in `tests/` and run with `pytest`.
- `python runner.py validate` is the integrity gate; keep it green.
- Before opening a PR, confirm the tests that do not need external resources
  pass, and note in the PR if you could not run the resource-dependent ones.

## Scale validity (read this before editing a task criterion)

Every scored criterion must be **discriminating**: a reply containing none of the
task's required content must not satisfy it. `tools/check_scoring_floor.py`
asserts this in CI, so a new permissive pattern fails the build rather than
quietly inflating every model's score.

When you change `data/elaboration_constraints.json`, a PR must contain all of:

```bash
# 1. the metric must still have a true zero
python tools/check_scoring_floor.py

# 2. and a reachable ceiling: add/update a positive probe in
#    tests/test_scoring_floor.py so the criterion cannot be "fixed" by
#    making it unsatisfiable

# 3. republish the deterministic baseline
python tools/run_smoke_baseline.py --output baselines/smoke-baseline.json

# 4. re-record the artifact digest, or `runner.py validate` fails
python - <<'PY'
import hashlib, json, pathlib
new = hashlib.sha256(pathlib.Path('data/elaboration_constraints.json').read_bytes()).hexdigest()
p = pathlib.Path('data/PROVENANCE.json'); s = p.read_text(encoding='utf-8')
old = json.loads(s)['artifacts']['data/elaboration_constraints.json']
p.write_text(s.replace(old, new, 1), encoding='utf-8')
PY

# 5. keep the tracked-file inventory byte-sorted (CI diffs it against git ls-files)
git add <new files> && git ls-files | LC_ALL=C sort > MANIFEST.txt
```

If the change alters what a score *means* rather than fixing a measurement bug,
that is a `BENCHMARK_VERSION` decision — see `docs/RELEASING.md`.

## Code style

- Python 3.11+.
- One import per concern, grouped: stdlib, third-party, local.
- Functions carry a docstring explaining *why* (design rationale), not just
  *what*.

## Licensing

New files should be Apache-2.0 unless they embed third-party content. Do not commit third-party dictionary/model binaries. Add them as immutable, checksum-verified managed resources and document their original licence and provenance in `docs/LICENSE-THIRD-PARTY.md`.
