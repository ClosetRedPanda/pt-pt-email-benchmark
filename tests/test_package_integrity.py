from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_support_directories_present():
    assert (ROOT / "models" / "README.md").is_file()
    assert (ROOT / "dics" / "README.md").is_file()
    assert (ROOT / "tools" / "review_wq_calibration.py").is_file()