from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_runs_the_full_cpu_safe_suite_without_broad_exclusions() -> None:
    workflow = _workflow_text()
    assert "python -m pytest -q" in workflow
    assert "pytest -q tests" not in workflow
    assert "--ignore" not in workflow
    assert "--deselect" not in workflow
    assert " -k " not in workflow
    assert "continue-on-error" not in workflow


def test_ci_installs_declared_test_extras_without_model_packages() -> None:
    workflow = _workflow_text()
    assert 'python -m pip install -e ".[dev,ocr]" build twine' in workflow
    assert "surya-ocr" not in workflow
    assert "chandra-ocr" not in workflow
    assert "torch" not in workflow
    assert "cuda" not in workflow.lower()


def test_ci_forces_hugging_face_runtime_offline() -> None:
    workflow = _workflow_text()
    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
    assert 'HF_HUB_DISABLE_TELEMETRY: "1"' in workflow


def test_ci_runs_static_type_and_package_checks() -> None:
    workflow = _workflow_text()
    assert "python -m ruff check ." in workflow
    assert "python -m mypy src scripts" in workflow
    assert "python -m build" in workflow
    assert "python -m twine check dist/*" in workflow


def test_ci_has_bounded_permissions_and_runtime() -> None:
    workflow = _workflow_text()
    assert "permissions:" in workflow
    assert "contents: read" in workflow
    assert "timeout-minutes: 20" in workflow
    assert "ubuntu-latest" in workflow
    assert 'python-version: "3.11"' in workflow
