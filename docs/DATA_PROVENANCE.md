# Data provenance

## Required disclosure

**All benchmark samples are LLM-assisted synthetic data.** They are not a
corpus of real customer email and must not be represented as naturally sampled
business correspondence. The tasks imitate realistic situations; that does not
make them real-world observations.

This disclosure applies to the source email/task samples. Labels, executable
constraints, calibration artifacts, and review files are derived benchmark
metadata; their current evidence and limitations remain described in
`VALIDATION.md`.

## What is known

- Dataset type: synthetic.
- LLM assistance: yes.
- Intended domain: PT-PT/English business-email understanding and generation.
- Current fixed sets: 20 structured-analysis samples and 20 PT-PT generation tasks.
- Exact artifact bytes: bound by SHA-256 in `data/PROVENANCE.json` and checked by
  `python runner.py validate`.

## Metadata gaps — do not guess

The original LLM provider/model/revision, generation prompts, generation dates,
selection process, and complete per-sample review log were not preserved in the
repository. They are intentionally `null` in `data/PROVENANCE.json`. A future
maintainer should fill those fields only from records, never from memory or
inference.

These gaps limit claims about contamination, diversity, prompt-induced style,
and independence. They are one reason this small benchmark supports pipeline
debugging and directional evidence, not broad population claims or tight model
rankings.

## Adding or changing samples

For each contribution, record the source type, generator/provider and exact
revision (if LLM-assisted), complete prompt/template, date, sampling/selection
method, licence, annotators/reviewers, and adjudication method. Update the
artifact digest in `data/PROVENANCE.json`; validation will fail until it matches.
Do not add confidential correspondence or personal data.
