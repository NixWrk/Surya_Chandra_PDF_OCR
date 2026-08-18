"""Focused offline checks for the deterministic synthetic benchmark recipe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.synthetic.v1.generate import generate_corpus, validate_corpus


EXPECTED_FIXTURES = {
    "blank-graphics",
    "clean-en",
    "clean-ru",
    "degraded-vector-text",
    "long-23p",
    "mixed-layout",
    "native-text-layer",
    "retention-3p",
    "rotated",
}


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generation_is_reproducible_and_model_free(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = generate_corpus(first)
    second_manifest = generate_corpus(second)

    assert first_manifest == second_manifest
    first_files = _files(first)
    second_files = _files(second)
    assert set(first_files) == set(second_files)
    for name, content in first_files.items():
        assert content == second_files[name], name
    assert first_manifest["provenance"] == {
        "contains_private_data": False,
        "external_assets": [],
        "generated_pdf_policy": "caller-output-only",
        "source_kind": "procedural-original",
    }
    assert first_manifest["metrics"]["model_status"] == "not_run"


def test_manifest_covers_fixture_matrix_and_case_regressions(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    manifest = generate_corpus(root)

    assert {fixture["id"] for fixture in manifest["fixtures"]} == EXPECTED_FIXTURES
    long_fixture = next(item for item in manifest["fixtures"] if item["id"] == "long-23p")
    assert long_fixture["page_count"] == 23
    assert long_fixture["chunk_plan"] == {"pages_per_chunk": 10, "expected_chunks": 3}

    punctuation = next(item for item in manifest["cases"] if item["id"] == "punctuation-only-chandra-v1")
    punctuation_case = json.loads((root / punctuation["path"]).read_text(encoding="utf-8"))
    assert punctuation_case["historical_expected"]["status"] == "error"
    assert punctuation_case["historical_expected"]["reason"] == "invalid_chandra_attempt_evidence"
    assert punctuation_case["current_expected"]["punctuation_lines_retained"] is True

    assert validate_corpus(root)["corpus_id"] == "uniscan-synthetic-offline"


def test_validation_rejects_tampered_source_pdf(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    generate_corpus(root)
    fixture = next(item for item in json.loads((root / "manifest.json").read_text(encoding="utf-8")[0:]) ["fixtures"] if item["id"] == "clean-en")
    pdf = root / fixture["source_pdf"]
    pdf.write_bytes(pdf.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="Source PDF hash mismatch: clean-en"):
        validate_corpus(root)


def test_manifest_hashes_are_sha256_and_cli_output_is_caller_scoped(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    manifest = generate_corpus(root)
    manifest_path = root / "manifest.json"

    assert manifest_path.is_file()
    assert all(path.is_relative_to(root) for path in root.rglob("*.pdf"))
    for fixture in manifest["fixtures"]:
        pdf = root / fixture["source_pdf"]
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        assert digest == fixture["source_pdf_sha256"]
