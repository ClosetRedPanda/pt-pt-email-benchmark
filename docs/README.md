# Managed Hunspell dictionaries (pt_PT / pt_BR)

The local dialect and writing-quality checks use dictionary pairs at
`docs/pt_PT.{aff,dic}` and `docs/pt_BR.{aff,dic}`. These files are **not
committed or redistributed** by this repository.

Install every managed evaluator resource with:

```bash
python runner.py setup
```

`core/resources.py` pins the LibreOffice source revision, expected byte count,
and SHA-256 for all four files. Evaluators verify the bytes before loading them.
If a pair is absent or corrupted, dictionary-backed metrics become unavailable
rather than a fabricated perfect score. See `LICENSE-THIRD-PARTY.md` for source
and licence details.
