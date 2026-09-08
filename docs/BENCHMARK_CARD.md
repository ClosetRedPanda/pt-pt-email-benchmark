# PT-PT Email LLM Benchmark — benchmark card

A short, structured summary so downstream users can judge what the scores do
and do not mean. Mirrors the "model card" convention for datasets/benchmarks.

## Intended use

- Comparing LLMs on **European Portuguese (PT-PT) business-email**
  understanding and generation.
- Favouriting **deterministic, reproducible** measurement over
  LLM-as-a-judge.
- Researchers/developers of Portuguese and multilingual LLMs.

## Not intended for

- Ranking models on **general Portuguese** or general language ability.
- Producing a single "overall quality" leaderboard number (deliberately not
  provided).
- Claims about formal inter-rater human agreement (not established; see
  `VALIDATION.md`).

## Task set (current)

| Task family | Reference size | Ground truth |
| --- | --- | --- |
| Structured analysis | 20 rows | Explicit ground truth in `data/analysis_reference.jsonl` |
| Generation (PT-PT elaboration) | 20 constraint sets | Executable deterministic checks in `data/elaboration_constraints.json` |

> **Caveat:** these sets are currently small. Scores are useful for debugging
> and directional signal, not yet for tight statistical ranking. Growing the
> sets is an open, high-priority task.

## Signals

- Structured understanding: schema validity, category/urgency/action/
  sentiment/language-variant accuracy, entity F1.
- Generation: instruction adherence, semantic preservation.
- PT-PT fidelity: EUPTVID probability, PT-PT compliance, PT-BR leakage,
  Word Fidelity (WF).
- Writing quality: `wq_defect_only_score` (primary) and calibrated
  `writing_quality_score`; dialect issues excluded to avoid double-penalisation.
- Performance: latency, throughput, tokens/sec, estimated cost.

## Determinism & audit

- No LLM judge on the scoring path.
- Raw outputs retained as JSONL.
- Runs write a hash-bound `*.manifest.json` sidecar (result/input/evaluator/
  resource hashes, parameters, command line).
- Legacy artifacts are immutable; rescoring is opt-in and labelled
  exploratory.

## Known limitations

- EUPTVID is a managed resource fetched at setup (not committed).
- Dictionary-backed signals depend on bundled Hunspell dictionaries; if
  absent they report unavailable, never a fabricated score.
- LanguageTool runs locally and requires a JRE.
- Human WQ reference is calibration evidence, not a passed multi-rater gate.
- Dictionaries are not dialect-matched siblings.

## Redistribution

See `LICENSE` (Apache-2.0). Bundled third-party assets retain their own
licences and provenance (see `docs/README.md`).
