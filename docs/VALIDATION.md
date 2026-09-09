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
- **Scoring-floor anchoring**: `tools/check_scoring_floor.py` (run in CI and on
  release) scores the shared no-content control against all 20 tasks and fails
  if any task-specific criterion passes. A criterion that boilerplate satisfies
  measures "is this an email", not adherence, so it is prohibited rather than
  documented as a caveat.
- Reproducibility of the scoring code from plain JSONL artifacts.
- New result artifacts have strict row validation and a hash-bound sidecar manifest.
- `compare.py` rejects missing manifests by default and rejects incompatible
	manifest metadata before comparison.

## Measured agreement of the leakage detector with the reference set
`python tools/gold_agreement.py` scores `evaluate_pt_dialect` against the
`reference_judgments` already present in `data/analysis_reference.jsonl`, and
`baselines/gold-agreement.json` freezes the result. It is measurement only: no
score on the ranking path changes, and the tool holds no vocabulary of its own.

It refuses to report rather than report a meaningless number. It exits non-zero
when the managed dictionaries are absent (the lexical-contrast path would not run
at all) and when the pinned `pt_core_news_sm` model is absent (non-lexical masking
would degrade). It also pins the SHA-256 of the two `.dic` files, which are
gitignored, so a dictionary update that changes a both-dictionaries verdict forces
a reviewed regeneration instead of passing as "same source, same numbers", and it
records `detector_engines` so a baseline can never be compared against a
differently-loaded detector.

The recorded state of the world:

- recall **66.67% (6/9)**, precision **100% (0 false alarms over 11 non-leaking
  rows)**, overall agreement **85%**.
- All three misses share one structural cause. Every marker the gold labels cites
  for them exists in *both* PT-PT and PT-BR dictionaries, and a rule whose test is
  "in one dictionary, not the other" cannot represent that case however its list
  grows. The miss is not a missing word; it is a limit of the mechanism.
- Headline precision depends on one heuristic, not on the detector being right.
  The rule fired on 10 of the 11 non-leaking rows; the capital-first-letter
  `score_eligible` rule discarded those findings. Without that rule precision is
  37.5%.
- This set cannot audit that heuristic. No non-leaking row carries a
  both-dictionaries marker, so there is nothing in it for a European false alarm
  to be found in. Absence of false positives here is not evidence of their absence.
- The tool also refuses to measure a half-installed environment, which matters
  because the failure is not obviously a failure: without the pinned
  `pt_core_news_sm` model, `_mask_nonlexical_spans` falls back to regex-only URI
  masking, `De:`/`Para:` address local-parts are read as prose, and this corpus
  gains five false alarms from personal names (`sofia`, `teresa`, `miguel`,
  `helena`, `pedro`), taking precision from 100% to 58.33%. That is the same rule
  their own `test_lexical_contrasts_mask_structured_nonlexical_spans` guards.
- EUPTVID agrees with gold on 90% (17/17 of the Portuguese rows) versus the rule's
  85%. It is reported for reference and is **not** a drop-in replacement: the
  roadmap forbids substituting its probability for compliance or WF.

## What is *not* claimed
The included WQ human reference contains 50 verified rows, but it does not establish multi-rater inter-rater reliability. The benchmark therefore does **not** call the formal human WQ gate passed.

EUPTVID and the Hunspell dictionaries are managed external assets: they are not committed to the repository. `python runner.py setup` fetches each from a pinned upstream revision and verifies its SHA-256 before use; every digest is recorded in run manifests. A resource that is absent or fails verification yields an unavailable signal rather than a fabricated one. Dataset files are likewise bound to the LLM-assisted synthetic provenance declaration by hashes in `data/PROVENANCE.json`.

Existing `v2` result files are legacy artifacts. They can be inspected only with
`--allow-legacy`; `--rescore` is explicitly labeled exploratory and does not yet
write a new immutable artifact. To persist one explicitly, use
`tools/rescore_artifact.py`, which writes a new JSONL file and a provenance
manifest containing the original artifact hash and evaluator source hashes.

## Philosophy
Raw outputs are retained. Automatic evaluators are instruments, not ground truth. There is no overall quality score in the primary result.
