# Hunspell dictionaries (pt_PT / pt_BR)

The local writing-quality and dialect checks in `core/pt_dialect.py` and
`core/writing_quality.py` look for Hunspell dictionary pairs at:

- `docs/pt_PT.dic` / `docs/pt_PT.aff`
- `docs/pt_BR.dic` / `docs/pt_BR.aff`

The repository stores these dictionary files under `docs/`. The `dics/`
directory remains as a compatibility and documentation location.

To enable these checks, obtain the official European/Brazilian Portuguese
Hunspell dictionary pairs (e.g. from LibreOffice's `pt_PT`/`pt_BR` dictionary
extensions) and place the four files above under `docs/` before running the
benchmark.
