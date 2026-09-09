# Hunspell compatibility directory

Dictionary files are no longer stored in Git or in this directory. The
benchmark retains its historical runtime paths under `docs/` for artifact
compatibility, but `python runner.py setup` now fetches immutable,
SHA-256-verified LibreOffice `pt_PT` and `pt_BR` pairs there.

See `docs/README.md` and `docs/LICENSE-THIRD-PARTY.md`. Do not commit downloaded
`.aff` or `.dic` files.
