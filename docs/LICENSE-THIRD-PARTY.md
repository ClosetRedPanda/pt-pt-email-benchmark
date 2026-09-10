# Third-party content and licences

This repository is Apache-2.0 (see `LICENSE`). Third-party evaluator resources
are **not committed**; `python runner.py setup` downloads exact revisions and
verifies every file before use.

## Hunspell dictionaries (`docs/pt_PT.*`, `docs/pt_BR.*`)

- Origin: LibreOffice `dictionaries`, directories `pt_PT` and `pt_BR`.
- Immutable revision: `1e848fbddd7fd8e03fb696ecc03ee1068fab141c`.
- Licence: GPL-2.0-or-later OR LGPL-2.1-or-later OR MPL-1.1 (not Apache-2.0).
- Distribution: downloaded on demand; ignored by Git; never included in the
  project source release.

The dictionaries are not dialect-matched siblings; coverage differences remain
a documented methodological limitation. Users who download them are
responsible for complying with their selected upstream licence.

## EUPTVID fastText classifier (`models/model_quantized.ftz`)

- Source: `duarteocarmo/fasttext-euptvid` (Hugging Face).
- Immutable revision: `f58c96b242d63b2c48f98cfb00045e4cf8cdf6b0`.
- Licence: MIT.
- Distribution: downloaded on demand and ignored by Git.

The exact SHA-256 and byte size of every managed file above are pinned in
`core/resources.py` and enforced by `runner.py setup` before any evaluator may
load them.
