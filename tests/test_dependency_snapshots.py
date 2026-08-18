from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / "constraints" / "observed"
SURYA = SNAPSHOT_ROOT / "docker-py311-cu126-surya.txt"
CHANDRA = SNAPSHOT_ROOT / "docker-py311-cu126-chandra.txt"
PROVENANCE = SNAPSHOT_ROOT / "README.md"
PIN = re.compile(r"(?P<name>[A-Za-z0-9_.-]+)==(?P<version>\S+)")


def _pins(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN.fullmatch(line)
        assert match is not None, f"Non-exact constraint in {path}: {line}"
        normalized = re.sub(r"[-_.]+", "-", match.group("name")).lower()
        assert normalized not in result
        result[normalized] = match.group("version")
    return result


def test_snapshots_are_exact_external_dependency_observations() -> None:
    for path in (SURYA, CHANDRA):
        text = path.read_text(encoding="utf-8")
        pins = _pins(path)
        assert pins
        assert " @ " not in text
        assert "file://" not in text
        assert "uniscan" not in pins


def test_observed_engine_profiles_are_recorded() -> None:
    surya = _pins(SURYA)
    chandra = _pins(CHANDRA)
    assert (surya["torch"], surya["surya-ocr"], surya["transformers"]) == (
        "2.11.0+cu126", "0.17.1", "4.57.1"
    )
    assert (chandra["torch"], chandra["chandra-ocr"], chandra["transformers"]) == (
        "2.11.0+cu126", "0.2.0", "5.14.1"
    )


def test_snapshot_is_explicitly_not_enforced() -> None:
    provenance = PROVENANCE.read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    setup = (ROOT / "setup_dual_venv.cmd").read_text(encoding="utf-8")
    assert "No clean Docker build has been validated" in provenance
    assert "docker-py311-cu126-surya.txt" not in dockerfile
    assert "docker-py311-cu126-chandra.txt" not in setup
    assert "${TORCH_CUDA_FLAVOR}" in dockerfile
    assert 'if /I "%TORCH_CUDA_FLAVOR%"=="cu128"' in setup
