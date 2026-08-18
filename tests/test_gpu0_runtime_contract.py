import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNTIME_CONTRACT_FILES = (
    ".env.example",
    "docker-compose.yml",
    "run_basic_gui.cmd",
    "setup_dual_venv.cmd",
    "scripts/benchmark_ocr_matrix.ps1",
    "scripts/gpu0_contract.ps1",
    "scripts/run_hybrid_gpu_smoke.ps1",
    "src/uniscan/ocr/benchmark.py",
)
MACHINE_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
    flags=re.IGNORECASE,
)
REQUIRED_COMPOSE_GPU_UUID = re.compile(
    r"\$\{UNISCAN_GPU_DEVICE_ID:\?[^}\r\n]+\}",
)


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_tracked_runtime_contract_has_no_machine_specific_gpu_uuid() -> None:
    occurrences: list[str] = []
    for relative in RUNTIME_CONTRACT_FILES:
        for line_number, line in enumerate(_read(relative).splitlines(), start=1):
            if MACHINE_GPU_UUID.search(line):
                occurrences.append(f"{relative}:{line_number}")

    assert occurrences == [], (
        "Host GPU UUIDs must be local configuration, not tracked runtime policy: "
        + ", ".join(occurrences)
    )


def test_compose_requires_one_host_gpu_uuid_and_exposes_only_container_gpu0() -> None:
    compose = _read("docker-compose.yml")
    environment_line = re.search(
        r"^\s*UNISCAN_GPU_DEVICE_ID:\s*[\"']?"
        r"(?P<value>\$\{UNISCAN_GPU_DEVICE_ID:\?[^}\r\n]+\})[\"']?\s*$",
        compose,
        flags=re.MULTILINE,
    )
    device_ids_line = re.search(
        r"^\s*device_ids:\s*\[\s*[\"']?"
        r"(?P<value>\$\{UNISCAN_GPU_DEVICE_ID:\?[^}\r\n]+\})[\"']?\s*\]\s*$",
        compose,
        flags=re.MULTILINE,
    )

    assert environment_line is not None, (
        "Compose must fail closed when UNISCAN_GPU_DEVICE_ID is unset in the "
        "container environment contract."
    )
    assert device_ids_line is not None, (
        "Compose must use the same required UNISCAN_GPU_DEVICE_ID expansion for "
        "the host device reservation."
    )
    assert environment_line.group("value") == device_ids_line.group("value")
    assert REQUIRED_COMPOSE_GPU_UUID.fullmatch(environment_line.group("value"))
    assert 'CUDA_VISIBLE_DEVICES: "0"' in compose
    assert "gpus: all" not in compose


def test_executable_gpu_contract_uses_environment_and_attests_index_zero() -> None:
    setup = _read("setup_dual_venv.cmd")
    helper = _read("scripts/gpu0_contract.ps1")
    gui = _read("run_basic_gui.cmd")
    smoke = _read("scripts/run_hybrid_gpu_smoke.ps1")
    matrix = _read("scripts/benchmark_ocr_matrix.ps1")

    assert "UNISCAN_GPU_DEVICE_ID" in setup
    assert "UNISCAN_GPU_DEVICE_ID" in helper
    assert "UNISCAN_GPU_DEVICE_ID" in gui
    assert "$env:UNISCAN_GPU_DEVICE_ID" in helper
    assert "nvidia-smi --id=0 --query-gpu=index,uuid" in setup
    assert "nvidia-smi --query-gpu" not in setup
    assert "& nvidia-smi --id=0" in helper
    assert '$env:CUDA_VISIBLE_DEVICES = "0"' in helper
    assert "Assert-UniscanGpu0Contract" in smoke
    assert "Assert-UniscanGpu0Contract" in matrix


def test_python_gpu_contract_reads_environment_and_attests_index_zero() -> None:
    benchmark = _read("src/uniscan/ocr/benchmark.py")

    assert 'os.environ.get("UNISCAN_GPU_DEVICE_ID")' in benchmark
    assert "_EXPECTED_GPU0_UUID" not in benchmark
    assert '"--id=0"' in benchmark
    assert '"--query-gpu=index,uuid"' in benchmark
