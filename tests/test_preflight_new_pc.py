from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_new_pc.ps1"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_preflight_has_explicit_targets_and_machine_readable_output() -> None:
    script = _script_text()
    assert '[ValidateSet("Windows", "Docker")]' in script
    assert "[switch]$Json" in script
    assert "ConvertTo-Json" in script
    assert "exit 1" in script


def test_preflight_checks_shared_gpu0_contract_without_discovery_fallback() -> None:
    script = _script_text()
    assert "Assert-UniscanGpu0Contract" in script
    assert '@("name", "driver_version", "compute_cap")' in script
    assert "UNISCAN_GPU_DEVICE_ID" not in script
    assert "nvidia-smi" not in script


def test_windows_preflight_checks_both_venvs_imports_and_cuda_tensor() -> None:
    script = _script_text()
    assert r".venv_chandra\Scripts\python.exe" in script
    assert r".venv_surya\Scripts\python.exe" in script
    assert 'module = "chandra"' in script
    assert 'module = "surya"' in script
    assert "import $module" in script
    assert "import uniscan" in script
    assert "import torch" in script
    assert "torch.cuda.is_available()" in script
    assert 'torch.ones((1,), device="cuda:0")' in script
    assert "& $environment.python -B -c $probe" in script


def test_docker_preflight_checks_config_service_daemon_and_external_network() -> None:
    script = _script_text()
    assert "docker version" in script
    assert "docker compose --project-directory" in script
    assert "config --services" in script
    assert '"ocr-api"' in script
    assert "docker network inspect zotero-automation" in script


def test_preflight_contains_no_mutating_or_ocr_commands() -> None:
    script = _script_text().lower()
    forbidden = (
        "new-item", "set-content", "add-content", "out-file", "copy-item",
        "move-item", "remove-item", "docker compose build", "docker compose up",
        "docker network create", "pip install", "setup_dual_venv", "searchable-pdf",
        "benchmark-ocr",
    )
    for token in forbidden:
        assert token not in script


def test_preflight_is_valid_powershell_syntax() -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is not available")
    escaped_path = str(SCRIPT).replace("'", "''")
    command = f"[scriptblock]::Create([IO.File]::ReadAllText('{escaped_path}')) | Out-Null"
    completed = subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
