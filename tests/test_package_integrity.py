from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_support_directories_present():
    assert (ROOT / "models" / "README.md").is_file()
    assert (ROOT / "dics" / "README.md").is_file()
    assert (ROOT / "tools" / "review_wq_calibration.py").is_file()

def test_release_and_provenance_metadata_present():
    import json
    provenance = json.loads((ROOT / "data" / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["dataset_type"] == "synthetic"
    assert provenance["llm_assisted"] is True
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert 'cff-version: "1.2.0"' in citation
    assert 'name: "ClosetRedPanda"' in citation
    assert (ROOT / "requirements.lock").is_file()
