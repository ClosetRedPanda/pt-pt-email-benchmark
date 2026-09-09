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

## Code style

- Python 3.11+.
- One import per concern, grouped: stdlib, third-party, local.
- Functions carry a docstring explaining *why* (design rationale), not just
  *what*.

## Licensing

New files should be Apache-2.0 unless they embed third-party content. Do not commit third-party dictionary/model binaries. Add them as immutable, checksum-verified managed resources and document their original licence and provenance in `docs/LICENSE-THIRD-PARTY.md`.
