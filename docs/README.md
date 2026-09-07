# Hunspell dictionaries (pt_PT / pt_BR)

The local writing-quality and dialect checks in `core/pt_dialect.py` and
`core/pt_dialect.py` and `core/writing_quality.py` look for Hunspell dictionary pairs at:

- `docs/pt_PT.dic` / `docs/pt_PT.aff`
- `docs/pt_BR.dic` / `docs/pt_BR.aff`

These binary dictionary files are intentionally **not bundled** in this
repository export (license/size reasons). Without them, `get_ptpt_dictionary()`
and `get_ptbr_dictionary()` return `None` and any dependent check degrades
gracefully rather than crashing or pretending a dictionary-backed check ran
with a perfect score.

To enable these checks, obtain the official European/Brazilian Portuguese
Hunspell dictionary pairs (e.g. from LibreOffice's `pt_PT`/`pt_BR` dictionary
extensions) and place the four files above in this directory before running
the benchmark.
