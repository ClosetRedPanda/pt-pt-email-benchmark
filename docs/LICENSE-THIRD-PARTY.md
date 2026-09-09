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

| Managed file | SHA-256 | Bytes |
| --- | --- | ---: |
| `docs/pt_PT.aff` | `975a209fcc892cb382fa5f34a28c391a39668661ce373ae071287809c5fcae24` | 95,089 |
| `docs/pt_PT.dic` | `e29ba2d7aa8a2ad43e9cb46ac6473064b661545c87002aea90e18899d98d3cc9` | 1,485,977 |
| `docs/pt_BR.aff` | `21d8ad2a769a60e17e2b5ea4ef11d4d593a58b9e2a82d642ef82d6a4c5523865` | 979,792 |
| `docs/pt_BR.dic` | `a38bfb26b68ece2834e79fe83e48d5792652970ace12db89d1b9674bf9933183` | 4,477,695 |

The dictionaries are not dialect-matched siblings; coverage differences remain
a documented methodological limitation. Users who download them are
responsible for complying with their selected upstream licence.

## EUPTVID fastText classifier (`models/model_quantized.ftz`)

- Source: `duarteocarmo/fasttext-euptvid` (Hugging Face).
- Immutable revision: `f58c96b242d63b2c48f98cfb00045e4cf8cdf6b0`.
- SHA-256: `00add97008d34b43803471daedb60d910de9b1eac15fc00814c336a4b23f0f6d`.
- Licence: MIT.
- Distribution: downloaded on demand and ignored by Git.
