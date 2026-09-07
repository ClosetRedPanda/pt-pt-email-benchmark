# Lean PT-PT Email LLM Benchmark

A deliberately small, auditable benchmark for Portuguese (European Portuguese / PT-PT) and English business-email models.

This is the **research benchmark core**, not a production job platform. It keeps the parts that answer the benchmark question and removes infrastructure that does not improve measurement: SQLite, web UI, resume/recovery manifests, storage-layer abstractions, scraped corpora, elaborate application QA, and generic concurrency torture tests.

## What it measures

### Structured understanding
Schema validity, category accuracy, urgency accuracy, action-required accuracy, sentiment accuracy, language-variant accuracy, and entity F1.

### Generation
Instruction adherence and semantic preservation using deterministic constraints extracted from the fixed task specification.

### PT-PT fidelity
EUPTVID probability (independent signal), PT-PT compliance, PT-BR leakage, and Word Fidelity (WF). EUPTVID is **not** the numerical base of WF.

### Writing quality
Deterministic local grammar, spelling, repetition, structural, style, and calibrated WQ signals. Dialect/regionalism issues are kept separate from WQ.

### Performance
Latency, throughput, tokens/sec, and known cost per 1,000 emails.

## Why it is smaller

The prior versions accumulated application-style machinery around the benchmark. That machinery can be useful in a hosted service but does not make the scientific measurement stronger. This package instead has one simple data flow:

`fixed inputs -> candidate model -> raw JSONL outputs -> deterministic evaluators -> transparent summary`

## Run

```bash
python -m pip install -r requirements.txt
python runner.py validate
```

For candidate evaluation:

```bash
export OPENROUTER_API_KEY="..."
python runner.py analysis --model "openai/gpt-4o-mini"
python runner.py generation --model "openai/gpt-4o-mini"
```

Compare existing runs:

```bash
# Manifest-backed artifacts are required for ranking/comparison.
python compare.py results/model_a.jsonl results/model_b.jsonl --kind generation
# Add --json when the comparison is consumed by another tool.
python compare.py results/model_a.jsonl results/model_b.jsonl --kind generation --json
# Add --full-dialect-checks to include slower LanguageTool dialect checks.
python compare.py results/model_a.jsonl results/model_b.jsonl --kind generation --full-dialect-checks
# Legacy artifacts require an explicit exploratory opt-in.
python compare.py results/legacy_a.jsonl results/legacy_b.jsonl --kind generation --allow-legacy
# Recompute missing fields in legacy artifacts; output is labeled exploratory.
python compare.py results/model_a.jsonl results/model_b.jsonl --kind generation --rescore
# Persist an exploratory rescore as a new manifest-backed artifact.
python tools/rescore_artifact.py results/legacy.jsonl results/rescored.jsonl
```

New runner outputs write a sidecar `*.manifest.json` containing the result hash,
benchmark/input/evaluator/resource hashes, run parameters, and command line.
Legacy JSONL files remain immutable and are not silently rescored.

## Data

`data/analysis_reference.jsonl` is the fixed analysis reference set.

`data/elaboration_prompts_pt_pt.json` and `data/elaboration_constraints.json` define the generation tasks and executable checks.

`data/wq_human_reference.jsonl` is retained as calibration evidence, not as a claim of independent multi-rater validation.

## Deliberate limitations

The trimmed distribution does not bundle the EUPTVID fastText model. Hunspell dictionaries are expected at `docs/pt_PT.*` and `docs/pt_BR.*`; without them, PT-PT compliance, PT-BR leakage, and WF remain unavailable. LanguageTool is local through `language_tool_python`.

Primary results should remain multidimensional. A single "overall quality" number is intentionally not produced.
