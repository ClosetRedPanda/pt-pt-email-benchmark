# Release, citation, and DOI checklist

The repository currently must not imply that a tag, GitHub release, or DOI
exists until the maintainer creates it. Use this checklist for the first and
subsequent releases.

## 1. Prepare metadata

1. Choose the software release version and update `project.version` in
   `pyproject.toml` only when needed.
2. **Do not automatically align `config.BENCHMARK_VERSION` with the package or
   release version.** It identifies score comparability, not publication. Keep
   `lean-1.1` while scoring semantics and fixed task data remain comparable;
   change it only after a deliberate compatibility review when scores or data
   change.
3. In `CITATION.cff`, optionally replace/supplement `ClosetRedPanda` in the
   clearly marked personal-identity section. Add the actual release `version`
   and `date-released`.
4. Review `.zenodo.json`, `docs/DATA_PROVENANCE.md`, and every remaining `null`
   provenance field. Fill only facts supported by records.
5. Update release notes with evaluator/data/resource changes and explicit
   non-claims. Never publish fabricated provider/model results.

## 2. Reproduce locally

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python runner.py setup
python runner.py validate
python -m pytest -q
python tools/run_smoke_baseline.py --output baselines/smoke-baseline.json
git diff --exit-code -- baselines/smoke-baseline.json
```

The smoke baseline is deterministic, makes no network/model call, and is
`ranking_eligible: false`. It proves only that the data/schema/constraint path
runs; it is not an LLM leaderboard result.

## 3. Tag and stage the GitHub release

```bash
VERSION=1.0.0
git status --short                 # must be clean
git tag -s "v$VERSION" -m "v$VERSION"  # use -a instead if signing is unavailable
git push origin "v$VERSION"
```

The tag workflow rechecks the project, regenerates the smoke baseline, and
creates a **draft** GitHub release with the baseline and citation metadata
attached. Review the draft and then publish it manually. Source patches cannot
create this remote release on their own.

## 4. Mint and record a DOI

1. Enable the GitHub repository in Zenodo before publishing the GitHub release.
2. Publish the reviewed GitHub release; wait for Zenodo to archive it.
3. Copy the minted version DOI and concept DOI from Zenodo. Do not invent one.
4. Add the appropriate DOI to `CITATION.cff` and the README, then make a small
   metadata follow-up release if necessary.
5. Cite the version DOI for a specific benchmark snapshot and archive each
   result JSONL together with its manifest.

A DOI identifies a release; it does not validate the benchmark methodology or
turn the smoke baseline into a model-ranking result.
