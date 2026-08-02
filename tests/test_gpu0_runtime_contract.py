from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_UUID = "GPU-e6a8c006-5017-6126-01cc-bf9bd972bf4f"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_compose_reserves_exact_host_gpu_and_exposes_only_container_gpu0() -> None:
    compose = _read("docker-compose.yml")

    assert f'UNISCAN_GPU_DEVICE_ID: "{EXPECTED_UUID}"' in compose
    assert 'CUDA_VISIBLE_DEVICES: "0"' in compose
    assert f'device_ids: ["{EXPECTED_UUID}"]' in compose
    assert "${UNISCAN_GPU_DEVICE_ID" not in compose
    assert "gpus: all" not in compose
    assert "UNISCAN_GPU_DEVICE_ID:-1" not in compose


def test_executable_gpu_probes_are_scoped_to_index_zero() -> None:
    setup = _read("setup_dual_venv.cmd")
    helper = _read("scripts/gpu0_contract.ps1")
    smoke = _read("scripts/run_hybrid_gpu_smoke.ps1")
    matrix = _read("scripts/benchmark_ocr_matrix.ps1")

    assert EXPECTED_UUID in setup
    assert "nvidia-smi --id=0 --query-gpu=index,uuid" in setup
    assert "nvidia-smi --query-gpu" not in setup
    assert "& nvidia-smi --id=0" in helper
    assert "Assert-UniscanGpu0Contract" in smoke
    assert "Assert-UniscanGpu0Contract" in matrix
    assert 'UNISCAN_OLMOCR_DOCKER_GPU = "device=$script:UniscanExpectedGpu0Uuid"' in matrix
