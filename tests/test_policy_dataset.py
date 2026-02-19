"""
Policy Engine dataset tests: run PolicyEngine against sample PDFs to verify accuracy.
Tune config/policies.yaml thresholds by running: pytest tests/test_policy_dataset.py -v
"""
import pytest
import sys
from pathlib import Path

# Project root and src on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Dataset paths (relative to repo root)
FAKE_DIR = ROOT / "tests" / "dataset" / "visa_refusals" / "fake"
REAL_DIR = ROOT / "tests" / "dataset" / "visa_refusals" / "real"
CONFIG_PATH = ROOT / "config" / "policies.yaml"


class MockUploadedFile:
    """File-like object compatible with analyze_file(uploaded_file)."""
    def __init__(self, bytes_data: bytes, name: str):
        self.bytes_data = bytes_data
        self.name = name

    def read(self):
        return self.bytes_data


@pytest.fixture(scope="module")
def policy_engine():
    """PolicyEngine using project config."""
    if not CONFIG_PATH.exists():
        pytest.skip(f"Config not found: {CONFIG_PATH}")
    from policy_engine import PolicyEngine
    return PolicyEngine(config_path=str(CONFIG_PATH))


def _pdf_paths(dir_path: Path):
    """List PDF paths in dir_path; dir_path may not exist."""
    if not dir_path.is_dir():
        return []
    return sorted(dir_path.glob("*.pdf"))


def test_visa_refusals_logic(policy_engine):
    """
    Run PolicyEngine on dataset PDFs:
    - fake/*.pdf -> verdict must be RED or AMBER
    - real/*.pdf -> verdict must be GREEN
    Tune policies.yaml and re-run pytest to verify thresholds.
    """
    from app import analyze_file

    fake_pdfs = _pdf_paths(FAKE_DIR)
    real_pdfs = _pdf_paths(REAL_DIR)

    if not fake_pdfs and not real_pdfs:
        pytest.skip(
            "No PDFs in dataset. Add PDFs to:\n"
            f"  {FAKE_DIR}\n"
            f"  {REAL_DIR}"
        )

    for pdf_path in fake_pdfs:
        bytes_data = pdf_path.read_bytes()
        mock_file = MockUploadedFile(bytes_data, pdf_path.name)
        analysis = analyze_file(mock_file)
        result = policy_engine.evaluate(analysis, "visa_refusal")
        verdict = result.get("verdict", "")
        assert verdict in (
            "RED",
            "AMBER",
        ), f"{pdf_path.name}: expected RED or AMBER for fake sample, got {verdict} (reason: {result.get('reason', '')})"

    for pdf_path in real_pdfs:
        bytes_data = pdf_path.read_bytes()
        mock_file = MockUploadedFile(bytes_data, pdf_path.name)
        analysis = analyze_file(mock_file)
        result = policy_engine.evaluate(analysis, "visa_refusal")
        verdict = result.get("verdict", "")
        assert verdict == "GREEN", (
            f"{pdf_path.name}: expected GREEN for real sample, got {verdict} "
            f"(reason: {result.get('reason', '')})"
        )
