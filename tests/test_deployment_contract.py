from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_default_compose_includes_the_ocr_service() -> None:
    compose = _read("docker-compose.yml")
    readme = _read("README.md")

    assert 'profiles: ["ocr"]' not in compose
    assert "docker compose up -d" in readme
    assert "docker compose --profile ocr up -d" not in readme


def test_documented_chandra_sidecar_default_is_strict() -> None:
    compose = _read("docker-compose.yml")
    readme = _read("README.md")

    assert "UNISCAN_CHANDRA_REQUIRE_SIDECAR=1" in readme
    assert 'UNISCAN_CHANDRA_REQUIRE_SIDECAR: "${UNISCAN_CHANDRA_REQUIRE_SIDECAR:-1}"' in compose
