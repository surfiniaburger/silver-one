import sys
import os
import pytest

# Ensure project root is on sys.path so tests can import local modules
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)


@pytest.fixture
def valid_review_payload():
    def build_payload(severity="OK"):
        return {
            "readability": {"score": 8, "rationale": "ok"},
            "maintainability": {"score": 8, "rationale": "ok"},
            "correctness": {"score": 8, "rationale": "ok"},
            "complexity": {"score": 8, "rationale": "ok"},
            "security": {"score": 8, "rationale": "ok"},
            "test_coverage": {"score": 8, "rationale": "ok"},
            "severity": severity,
            "summary": "legacy review",
            "findings": [],
        }

    return build_payload
