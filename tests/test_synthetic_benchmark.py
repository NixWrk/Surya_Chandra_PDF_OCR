"""Focused offline checks for the deterministic synthetic benchmark recipe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.synthetic.v1.generate import generate_corpus, validate_corpus
from benchmarks.synthetic.v1.evaluate import evaluate_run, main as evaluate_main, validate_run_input


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

def _measurements() -> dict[str, object]:
    return {
        "stage_timing_seconds": {
            "surya": "not_run",
            "chandra": "not_run",
            "pdf_build": "not_run",
            "validation": "not_run",
        },
        "peak_ram_bytes": "not_measured",
        "peak_vram_bytes": "not_measured",
        "engine_invocations": "not_run",
        "render_count": "not_run",
        "chunk_reuse": {
            "status": "not_run",
            "reused_chunks": "not_run",
            "rerun_reasons": "not_run",
        },
    }


def _run_record(root: Path, fixture_id: str, pages: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "uniscan.synthetic-evaluation-input.v1",
        "run_id": "offline-contract-v1",
        "model_status": "not_run",
        "fixtures": [{"fixture_id": fixture_id, "pages": pages}],
        "measurements": _measurements(),
        "output": {
            "status": "not_run",
            "path": "not_run",
            "sha256": "not_run",
            "bytes": "not_measured",
            "page_count": "not_measured",
        },
    }


def _exact_pages(root: Path, fixture_id: str) -> list[dict[str, object]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fixture = next(item for item in manifest["fixtures"] if item["id"] == fixture_id)
    rows = [
        json.loads(line)
        for line in (root / fixture["ground_truth"]).read_text(encoding="utf-8").splitlines()
    ]
    return [
        {
            "source_page": row["page"],
            "output_page": row["page"],
            "searchable_text": row["text"],
            "outcome": row["expected_outcome"],
        }
        for row in rows
    ]


def test_model_free_evaluator_is_deterministic_and_exact(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    generate_corpus(root)
    run = _run_record(root, "clean-en", _exact_pages(root, "clean-en"))

    first = evaluate_run(root, run)
    second = evaluate_run(root, run)

    assert first == second
    assert first["evaluator"]["model_invoked"] is False
    assert first["evaluator"]["output_metadata"] == "externally_attested_not_verified"
    assert first["fixtures"][0]["metrics"]["cer"]["value"] == 0.0
    assert first["fixtures"][0]["metrics"]["wer"]["value"] == 0.0

def test_accepted_zero_output_is_a_valid_page_outcome(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    generate_corpus(root)
    pages = _exact_pages(root, "clean-en")
    pages[0]["outcome"] = "accepted_zero_output"

    report = evaluate_run(root, _run_record(root, "clean-en", pages))

    assert report["fixtures"][0]["page_outcomes"]["observed"] == {"accepted_zero_output": 1}
    assert report["fixtures"][0]["metrics"]["cer"]["value"] == 0.0

    assert first["summary"]["searchability"]["exact_text_retention"]["status"] == "pass"
    assert first["summary"]["page_mapping"]["status"] == "pass"


def test_evaluator_reports_text_and_page_mapping_failures(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    generate_corpus(root)
    pages = _exact_pages(root, "retention-3p")
    pages[1]["searchable_text"] = "WRONG PAGE THREE"
    pages[1]["output_page"] = 3
    pages.pop()
    pages.append(
        {
            "source_page": 4,
            "output_page": 4,
            "searchable_text": "unexpected",
            "outcome": "text",
        }
    )

    report = evaluate_run(root, _run_record(root, "retention-3p", pages))
    fixture = report["fixtures"][0]

    assert fixture["metrics"]["cer"]["value"] > 0.0
    assert fixture["searchability"]["exact_text_retention"]["status"] == "fail"
    assert fixture["page_mapping"]["status"] == "fail"
    assert fixture["page_mapping"]["missing_source_pages"] == [3]
    assert fixture["page_mapping"]["unexpected_source_pages"] == [4]
    assert fixture["page_mapping"]["wrong_output_pages"] == [2]
    assert fixture["page_mapping"]["extra_output_pages"] == [4]
    assert fixture["page_outcomes"]["not_run_pages"] == [3]


def test_measurement_contract_requires_sentinels_or_nonnegative_values(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    generate_corpus(root)
    run = _run_record(root, "clean-en", _exact_pages(root, "clean-en"))

    missing = json.loads(json.dumps(run))
    del missing["measurements"]["peak_vram_bytes"]
    with pytest.raises(ValueError, match="measurements is missing peak_vram_bytes"):
        validate_run_input(missing)

    invalid = json.loads(json.dumps(run))
    invalid["measurements"]["peak_ram_bytes"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_run_input(invalid)

    invalid["measurements"]["peak_ram_bytes"] = "not_measured"
    invalid["output"]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_run_input(invalid)


def test_evaluator_cli_writes_canonical_repeatable_report(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    generate_corpus(root)
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps(_run_record(root, "clean-en", _exact_pages(root, "clean-en")), ensure_ascii=False),
        encoding="utf-8",
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    evaluate_main(["--corpus", str(root), "--run", str(run_path), "--output", str(first_path)])
    evaluate_main(["--corpus", str(root), "--run", str(run_path), "--output", str(second_path)])

    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text(encoding="utf-8"))["schema"] == (
        "uniscan.synthetic-evaluation-report.v1"
    )
