# PT-PT Email LLM Benchmark

**An open-source benchmark for evaluating large language models on European Portuguese (PT-PT) business email understanding and generation.**

The **PT-PT Email LLM Benchmark** evaluates how well language models handle real-world business-email tasks in European Portuguese, with particular attention to **PT-PT linguistic fidelity, PT-BR leakage, structured email understanding, generation quality, and deterministic evaluation**.

Unlike benchmarks that rely primarily on LLM-as-a-judge scoring, this project uses **transparent, deterministic evaluation methods** that can be reproduced locally and audited.

The benchmark measures:

* European Portuguese (PT-PT) language fidelity
* PT-PT vs PT-BR distinction
* Email classification and structured understanding
* Instruction adherence and semantic preservation
* Grammar, spelling, repetition, and writing quality
* Latency, throughput, tokens/second, and estimated cost

The benchmark is designed for researchers and developers evaluating **Portuguese LLMs, multilingual language models, email-generation models, and European Portuguese NLP systems**.

## What it measures (In detail)

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
# Required by the PT-PT grammar rules (spaCy model is not a pip dependency):
python -m spacy download pt_core_news_sm
# Required by the Hunspell spelling fallback:
python -m pip install spylls
python runner.py validate
```

### External requirements

| Requirement | Needed for | If missing |
| --- | --- | --- |
| `pt_core_news_sm` (spaCy) | PT-PT grammar rules (proclisis, `ter`/`haver`, gerund) | Those rules are skipped; dialect signals lose grammar evidence |
| `spylls` | Hunspell dictionary loading | PT-PT compliance, PT-BR leakage and WF report as unavailable |
| Java runtime (JRE 8+) | `language_tool_python` local server | LanguageTool grammar checks are skipped |
| `docs/pt_PT.*`, `docs/pt_BR.*` | Hunspell dictionaries (bundled) | As above |

Pin the spaCy model version alongside the package: dependency labels and
morphological features can change between model releases, and the PT-PT
grammar rules read those features directly, so an unpinned model can move
compliance results without any code change. The installed version is recorded
in each run manifest.

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
