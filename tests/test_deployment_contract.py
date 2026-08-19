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


def test_default_compose_is_standalone_without_a_precreated_external_network() -> None:
    compose = _read("docker-compose.yml")

    assert "zotero-automation" not in compose
    assert "external: true" not in compose


def test_shared_network_integration_is_an_explicit_override() -> None:
    override = _read("docker-compose.shared-network.yml")

    assert 'name: "${UNISCAN_SHARED_NETWORK_NAME:-zotero-automation}"' in override
    assert "external: true" in override


def test_compose_keeps_jobs_on_bind_and_uses_native_volume_for_chunk_cache() -> None:
    compose = _read("docker-compose.yml")
    architecture = _read("docs/ARCHITECTURE.md")
    deployment = _read("docs/runbooks/NEW_PC_DEPLOYMENT.md")

    assert "- ./outputs:/data/work" in compose
    assert "- hybrid-chunk-cache:/data/work/runs/hybrid_chunk_cache" in compose
    assert "hybrid-chunk-cache:" in compose
    assert (
        'name: "${UNISCAN_HYBRID_CACHE_VOLUME:-surya-chandra-ocr-hybrid-chunk-cache}"'
        in compose
    )
    assert "Docker-managed volume" in architecture
    assert "UNISCAN_HYBRID_CACHE_VOLUME" in deployment
