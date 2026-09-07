# Lean design: what survived the cut

## Kept because it changes benchmark validity

- Strict structured-output schema and conservative parsing.
- Deterministic instruction-adherence checks.
- Deterministic semantic-preservation checks.
- Separate PT-PT classifier, compliance, PT-BR leakage, and WF signals.
- Local deterministic writing-quality engine with a frozen calibration artifact.
- Raw output retention in JSONL.
- Explicit latency, token, and known-cost recording.
- Honest `None`/unavailable states instead of silently converting missing evidence to a score.

## Removed because it was infrastructure, not measurement

- SQLite persistence and query/export layers.
- Web dashboard and static UI.
- Crash recovery, resume manifests, file locks, and dataset-recovery paths.
- Scraped `good_emails` / `bad_emails` corpora.
- Production-provider lifecycle checks and pricing/storage test matrix.
- Hundreds of application QA tests duplicating behavior that does not change a benchmark claim.
- Fabricated agreement fixtures and historical work artifacts.
- An aggregate “overall quality” leaderboard score.

## Result

The benchmark has one auditable path:

`fixed reference/task -> candidate model -> raw JSONL -> deterministic scoring -> multidimensional comparison`

That is the smallest form that still preserves the core scientific claims from the larger versions.
