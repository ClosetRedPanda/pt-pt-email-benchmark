# Hunspell dictionaries (pt_PT / pt_BR)

The local writing-quality and dialect checks in `core/pt_dialect.py` and
`core/writing_quality.py` use the Hunspell dictionary pairs at:

- `docs/pt_PT.dic` / `docs/pt_PT.aff`
- `docs/pt_BR.dic` / `docs/pt_BR.aff`

These files are committed in this repository and are read directly by the
benchmark (see the "bundled" row in the README external-requirements table).
`docs/pt_BR` is the reference PT-BR side for the leakage/contrast checks; the
two dictionaries are not dialect-matched siblings.

If you redistribute the project, keep the original licence and provenance
notice of the dictionary files with them and confirm it is compatible with
your target distribution licence.

If the files are absent or unreadable, `get_ptpt_dictionary()` and
`get_ptbr_dictionary()` return `None` and the dictionary-backed checks
degrade gracefully rather than reporting a fabricated perfect score.
