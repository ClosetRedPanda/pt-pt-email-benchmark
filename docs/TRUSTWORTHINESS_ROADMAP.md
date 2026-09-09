# Benchmark Trustworthiness Roadmap

## Goal

Make the benchmark usable for repeatable model comparisons and reliable enough that every reported score has:

- a known definition;
- a known denominator;
- reproducible inputs and evaluator versions;
- explicit treatment of failures and unavailable resources;
- validation against independent evidence.

The benchmark should prefer `N/A` and an explanation over a plausible but unsupported number.

## Current Position

The current benchmark is useful for exploration, but existing result files are not yet a clean scientific comparison.

Known limitations:

- Older artifacts were generated before later evaluator and resource-path fixes.
- `compare.py` can explicitly rescore legacy content, but rescored and persisted metrics must not be silently mixed.
- Provider failures have previously appeared as apparently successful rows.
- Placeholder detection was previously too narrow and required correction.
- Instruction adherence and semantic preservation are deterministic pattern checks, not general language understanding or human judgments.
- PT-PT metrics depend on Hunspell resources and evaluator versions.
- Writing quality is a deterministic proxy and does not yet have demonstrated multi-rater reliability.
- Latency, throughput, token counts, and cost can include different provider and evaluator effects.

### Already implemented

- API responses with provider errors or unsuccessful finish reasons are rejected by the current client.
- New runner records include explicit `status` values.
- Failed records are excluded from generation quality and performance aggregates.
- Missing, duplicate, failed, and unexpected analysis results are reported.
- Hunspell loaders use the dictionaries stored at `docs/pt_PT.*` and `docs/pt_BR.*`.
- Normal `compare.py` runs use stored artifact metrics only.
- Legacy recomputation is explicit through `--rescore`.
- Generic placeholder detection and Markdown-link exclusions have regression tests.
- Result rows are validated for identity, status, numeric values, and expected task IDs.
- Strict generation validation rejects empty, refused, provider-error, and truncated outputs.
- Manifest-backed artifacts bind result hashes to benchmark inputs, evaluator source,
  optional resources, and installed dependency versions.
- Scorecards expose metric-specific denominators and explicitly list unavailable metrics.
- Comparisons provide deterministic paired task-level bootstrap intervals, wins,
  losses, and ties without treating independent task samples as independent runs.
- Mutation tests cover negation, changed facts, forbidden changes, placeholders,
  and Markdown-link false positives.

### Still unsafe or incomplete

- Existing `v2` artifacts are legacy artifacts and were not created with the current evaluator contract.
- `--rescore` is exploratory; use `tools/rescore_artifact.py` to persist a new
	manifest-backed artifact.
- Existing `v2` artifacts remain legacy artifacts without manifests.
- A manifest records package versions, but external service behavior and provider
	model revisions remain outside local control.
- Paired bootstrap comparison is available, but the task set remains small and the
	interval reflects uncertainty across these tasks rather than population validity.
- The WQ calibration is not demonstrated to have independent multi-rater reliability.
- Instruction and semantic checks remain regex/pattern-based proxies. The
  committed constant smoke baseline scores 13.33% mean adherence and reaches
  50–66.7% on some tasks without task-specific content, directly demonstrating
  a permissive-pattern floor that must be fixed before close model ranking.
- The result validator and manifest are versioned, but the row contract is not yet
	a complete formal schema for every optional provider field.

## Priority 1: Freeze Reproducibility

### 1. Define a run identity

Every run should record:

- benchmark version;
- run ID and timestamp;
- model ID and provider/route;
- system prompt hash;
- prompt-data hash;
- constraint-file hash;
- evaluator version for every evaluator;
- temperature, maximum tokens, timeout, retry settings, and concurrency;
- Python version and dependency versions;
- hashes and availability of Hunspell, EUPTVID, and LanguageTool resources.

A result must be reproducible from its manifest without relying on the current state of the repository.

### 2. Freeze resources

Pin or hash:

- `docs/pt_PT.dic` and `docs/pt_PT.aff`;
- `docs/pt_BR.dic` and `docs/pt_BR.aff`;
- the EUPTVID model;
- LanguageTool version and language rules;
- calibration artifacts;
- benchmark source and data files.

Resource changes must create a new benchmark/evaluator version.

### 3. Stop silent legacy rescoring

Normal comparison should use only metrics persisted in the artifact. Rescoring must be explicit and visibly labeled with:

- original artifact identity;
- new evaluator version;
- new resource hashes;
- rescore timestamp;
- changed metrics.

Do not rank a persisted run against a silently rescored run.

The current safe commands are:

```bash
# Compare only values persisted in the artifacts.
python compare.py results/a.jsonl results/b.jsonl --kind generation

# Recompute legacy fields explicitly. This is exploratory, not a final ranking.
python compare.py results/a.jsonl results/b.jsonl --kind generation --rescore
```

## Priority 2: Make Result Artifacts Strict

### 4. Define and validate a result schema

Each row should contain, where applicable:

- unique task ID;
- model ID;
- `status`: `success` or `error`;
- content or structured response;
- finish reason;
- provider error details;
- request start/end timestamps;
- provider latency;
- evaluator latency;
- prompt, completion, reasoning, and total token counts;
- provider-reported and catalog-estimated cost;
- evaluator outputs and evaluator versions.

Reject or clearly quarantine rows with:

- duplicate IDs;
- missing IDs;
- unexpected IDs;
- missing status;
- provider error metadata;
- refusal or unsuccessful finish reasons;
- empty or truncated output;
- non-finite or negative numeric values.

### 5. Separate requested, attempted, and successful samples

Every report should show:

- requested samples;
- attempted samples;
- successful samples;
- failed samples;
- missing samples;
- duplicate/unexpected samples;
- denominator for every metric.

Failures must never improve quality or performance scores.

For every averaged metric, report both the value and its denominator. For
example, `semantic preservation: 91.1% over 15/20 constrained tasks` is
meaningful; `semantic preservation: 91.1%` alone is ambiguous.

### 6. Preserve raw evidence

Keep the raw provider response, but distinguish clearly between:

- raw provider output;
- accepted model content;
- rejected/failed provider output;
- repaired structured output;
- evaluator-derived values.

A repaired response must not look identical to a clean valid response.

## Priority 3: Improve Metric Definitions

### 7. Instruction adherence

Report two views:

- per-task score, where every task has equal weight;
- criterion-weighted score, where every executable criterion has equal weight.

Also report:

- passed criteria;
- failed criteria;
- placeholder failures;
- denominator and task IDs for failures.

Document which view is primary before comparing models.

Do not interpret this metric as general helpfulness, tone quality, or factual
correctness. It measures whether configured executable patterns were detected.

### 8. Placeholder detection

Keep the corrected generic placeholder detection and test it against a locked fixture set containing:

- `[Seu Nome]`;
- `[Empresa]`;
- `[Contato]`;
- `[Nome do Cliente]`;
- `[inserir endereço]`;
- `[XXXX-XX-XX]`;
- empty or partial template fields;
- Markdown links, which must not be treated as placeholders.

Review false positives involving ordinary bracketed prose, citations, URLs, and product notation.

**Policy (why unfilled placeholders count as failures).** Detection of leftover
template placeholders is a deliberate and strict instruction-adherence check, not an
evaluator defect. When a task asks the model to address a specific recipient, company,
or date, the prompt is unambiguously the source of truth; an unfilled `[Nome do
Cliente]` means the output is not ready to send, however polished the surrounding
prose. Counted placeholders therefore lower `instruction_adherence_pct` by design, and
any unresolved `[bracket]` occurrence is reported as a failed criterion rather than
silently tolerated. Two corollaries follow. First, the metric is scoped to bracketed
template slots only: Markdown links, citations, and ordinary bracketed prose are
masked so that legitimate content is never mistaken for an unfilled slot. Second, the
correct response to a placeholder failure is prompt-side remediation (instruct the
model to fill every field, or substitute the supplied entity), not weakening the
detector; regression suites should keep the locked fixture set above to guarantee that
detection stays strict for genuine slots and quiet for the masked look-alikes.

### 9. Semantic preservation

Label the current metric accurately as deterministic fact-pattern coverage. It is not a general semantic or hallucination score.

Report:

- constrained-task denominator;
- required facts passed/missed;
- forbidden facts detected;
- per-task details;
- unconstrained tasks separately.

An unconstrained task is not a failed semantic task and must not be silently
included as a zero. It should be counted separately from the constrained
denominator.

The current 20-task set defines no `forbidden_changes`, so the contradiction /
invented-fact half of the semantic evaluator is implemented but never exercised
by these tasks; until tasks exercise it, the metric can only report missing
required facts, not hallucinated ones.

Add adversarial tests for:

- negation;
- double negation;
- changed dates;
- changed quantities/prices;
- similar but incorrect identifiers;
- contradictions;
- invented facts;
- facts mentioned only in a disclaimer.

### 10. PT-PT fidelity

Keep these as separate metrics:

- EUPTVID classifier probability;
- PT-PT compliance;
- PT-BR leakage rate;
- validated violation count;
- diagnostic candidate count;
- Word Fidelity.

Never use EUPTVID probability as a substitute for compliance or WF.

Every report should include:

- dictionary/model hashes;
- number of scored rows;
- number of validated violations;
- number of diagnostic-only findings;
- whether the metric is full, degraded, or unavailable.

Build a balanced expert-labeled test set containing:

- authentic PT-PT;
- authentic PT-BR;
- mixed text;
- English output;
- technical loanwords;
- proper names;
- valid regional variants;
- dictionary coverage gaps.

Measure false positives and false negatives before using the metric for ranking.

The EUPTVID probability is a classifier output, not a calibrated probability
of perfect PT-PT writing. A high EUPTVID value cannot cancel a detected PT-BR
lexical or grammatical violation.

Documented limitations of the lexical-contrast layer (as of the 2026-09
release, all by design):

- the two managed dictionaries are not dialect-matched siblings: pt_BR derives
  compounds (e.g. `intranet` via `intra-` + `net`) that pt_PT.dic does not
  cover, so a correct business word absent from pt_PT.dic reads as a PT-BR
  leak. This is the dictionary-coverage-gap class; it is out of scope for the
  no-hardcoded-exceptions rule and should be revisited by honouring the
  dictionaries' own derivation metadata (affix rules, `PREAO90=` conversions)
  rather than word lists;
- the existential-`ter` rule does not detect plural-quantifier objects
  (`Tem vários erros…`): `pt_core_news_sm` tags `vários` as NOUN/amod, so the
  indefinite-determiner scan never fires. Known deterministic false negative;
- `Word fidelity` is leak *density* (weighted penalties ÷ words): a single leak
  costs less in a longer email, so it is reported with a density label and must
  not be read as a percentage of clean mail;
- the spelling sub-stage of writing quality accepts a word present in either
  bundled Portuguese dictionary (pt_PT or pt_BR); dialect judgement stays in
  the dialect layer so one incomplete dictionary cannot cause spelling false
  positives;
- task-echoed vocabulary is exempt from the lexical contrast: when a task's own
  required-action/fact patterns supply a word as an accepted answer (e.g.
  `estorno`), echoing it is the model following the benchmark, not a
  model-initiated dialect leak, so the runner and rescorer pass the task's
  pattern vocabulary as `echo_vocab` to the dialect evaluator. Standalone
  evaluator calls without task context remain strict.

### 11. Writing quality

Report dimensions separately:

- grammar;
- spelling;
- repetition;
- structure;
- style/register.

Treat the aggregate as a deterministic proxy until human validation demonstrates reliability.

Run sensitivity analyses with and without:

- placeholders;
- missing signatures;
- Markdown formatting;
- email headers;
- LanguageTool style warnings.

Use blinded multi-rater human labels with separate calibration and test sets. Report:

- number of raters;
- agreement statistics;
- confidence intervals;
- calibration/test split;
- known disagreements and limitations.

Do not report the aggregate WQ number as human quality until the human-rating
study is complete. Preserve the individual dimensions so a model cannot appear
better merely because it avoids one class of detector finding.

Grammar evidence depends on a responding local LanguageTool server: when none
responded, `grammar_errors_per_email` is reported as unavailable (N/A), never as
a measured zero, and the report states that grammar evidence was unavailable.

The writing-quality evaluator version is surfaced per scorecard (`wq_evaluator_version`);
when an artifact mixes rows from different evaluator versions the field reports
`mixed` instead of silently attributing everything to the first row's version.

## Priority 4: Make Performance Comparable

### 12. Separate timing categories

Record and report separately:

- provider request latency;
- retry and backoff time;
- rate-limit wait time;
- local dialect evaluation time;
- local WQ evaluation time;
- total wall-clock time.

Throughput should state whether it includes evaluator work. Compare runs only when concurrency and timing definitions match.

### 13. Normalize token and cost accounting

Report separately:

- prompt tokens;
- visible completion tokens;
- reasoning tokens;
- total provider tokens;
- provider-reported cost;
- catalog-estimated cost;
- unknown cost samples.

Do not compare tokens/sec when one provider reports hidden reasoning tokens and another does not, unless the definition is normalized.

At minimum, publish visible completion tokens/sec separately from provider
total-token throughput. Cost comparisons must state whether the value is
catalog-estimated or provider-reported.

## Priority 5: Statistical Reliability

### 14. Use paired comparisons

The same task IDs should be compared row by row. Report:

- per-task differences;
- mean and median differences;
- wins, losses, and ties;
- bootstrap confidence intervals;
- uncertainty caused by failed samples.

Do not declare a model better when the observed difference is smaller than the uncertainty interval.

### 15. Avoid overinterpreting small datasets

The current 20 generation tasks are useful for smoke testing and exploratory comparison, but not enough for strong general claims.

Expand the task set across:

- customer support;
- billing;
- logistics;
- sales;
- internal communication;
- technical support;
- formal and semi-formal registers;
- short and long responses;
- direct and indirect requests.

Keep a frozen test set separate from development and calibration data.

## Priority 6: Operational Safeguards

### 16. Prevent accidental data loss

- Require model/run-specific output paths, or refuse overwrite.
- Store a manifest beside every result file.
- Store the generated scorecard beside the raw JSONL.
- Keep legacy artifacts immutable.
- Record the command line used for the run.

### 17. Add adversarial regression tests

At minimum, cover:

- provider error responses;
- refusal responses;
- empty output;
- truncation;
- missing/duplicate/unexpected IDs;
- missing cost;
- unknown resources;
- generic placeholders;
- Markdown links;
- proper names in dialect dictionaries;
- PT-BR lexical terms;
- negation and changed numbers;
- scorecard denominator behavior.

### 18. Make validation fail loudly

`validate` should fail when:

- resources are expected but missing;
- reference IDs are duplicated;
- prompt and constraint IDs differ;
- result schemas are invalid;
- benchmark version metadata is inconsistent;
- evaluator versions cannot be identified.

## Recommended Execution Order

1. Freeze and hash benchmark inputs, evaluators, dependencies, and resources.
2. Add strict result-artifact validation and status handling.
3. Make all metric denominators visible.
4. Keep normal comparison stored-only; make rescoring explicit and traceable.
5. Finish placeholder and adversarial evaluator tests.
6. Rerun or rescore existing artifacts consistently under one pinned environment.
7. Separate provider, evaluator, retry, and wall-clock timing.
8. Add paired comparisons and bootstrap uncertainty intervals.
9. Validate dialect and WQ metrics against independent human labels.
10. Expand the dataset only after the measurement contract is stable.

## Immediate Next Work

Do these in order before changing evaluator heuristics again:

1. Add a strict result-artifact validator and make `compare.py` reject invalid or mixed artifacts.
2. Add a manifest writer containing input hashes, evaluator versions, resource hashes, command line, and run parameters.
3. Add metric denominators and scored-row counts to the scorecard/report.
4. Add a persisted rescore command that writes a new artifact and provenance record instead of only changing values in memory.
5. Add paired task-level comparison output and bootstrap confidence intervals.
6. Treat `gpt4o_mini_generation_v2.jsonl` and `qwen3-14b_generation_v2.jsonl` as exploratory legacy evidence until steps 1-4 are complete.

Do not spend time expanding the dataset or tuning WQ penalties before these
controls exist. Better-looking scores are not useful if their provenance and
denominators cannot be checked.

## Definition of Done

The benchmark is ready for serious model comparisons when:

- every result has a complete manifest;
- every row has an unambiguous status;
- failed calls cannot affect quality metrics;
- every average shows its denominator;
- old artifacts cannot be silently rescored;
- identical inputs and versions produce identical deterministic results;
- evaluator outputs have adversarial regression coverage;
- PT-PT and WQ metrics have independent validation evidence;
- performance and cost definitions are comparable;
- model differences include uncertainty, not only point estimates.
