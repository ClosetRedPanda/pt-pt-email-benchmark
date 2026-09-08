# Third-party content & licences

This repository is Apache-2.0 (see `LICENSE`). Files imported from other
projects keep their own licences; this page records where they come from and
what to check before redistribution.

## Hunspell dictionaries (`docs/pt_PT.*`, `docs/pt_BR.*`)

- **Origin:** LibreOffice's `pt_PT` and `pt_BR` Hunspell dictionary extensions.
- **Licence:** the LibreOffice/MySpell dictionary tri-licence —
  GPL-2.0-or-later OR LGPL-2.1-or-later OR MPL-1.1 — **not** Apache-2.0.
- **Why committed:** they are read directly by the offline lexical/dialect
  checks in `core/pt_dialect.py` and `core/writing_quality.py`.
- **Action before publishing the package:** confirm that redistributing these
  dictionaries under their own terms is acceptable for your intended
  distribution. If it is not, remove them and switch the dependent checks to a
  managed resource fetched at setup time (the same mechanism `core/resources.py`
  uses for the EUPTVID model) so they report *unavailable* rather than bundling
  the files. This is an explicit, unresolved decision for the maintainer.

## EUPTVID fastText classifier (`models/model_quantized.ftz`)

- **Source:** `duarteocarmo/fasttext-euptvid` (Hugging Face).
- **Licence:** MIT.
- **Distribution:** not committed; fetched and SHA-256 verified by
  `python runner.py setup` (see `models/README.md`).
