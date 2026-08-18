#!/usr/bin/env python3
"""Evaluate synthetic benchmark outputs without invoking OCR engines or models."""
from __future__ import annotations
import argparse
import copy
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.synthetic.v1.generate import validate_corpus


INPUT_SCHEMA = "uniscan.synthetic-evaluation-input.v1"
REPORT_SCHEMA = "uniscan.synthetic-evaluation-report.v1"
EVALUATOR_VERSION = "1.0.0"
SENTINELS = frozenset({"not_measured", "not_run"})
STAGES = ("surya", "chandra", "pdf_build", "validation")
SUCCESS_OUTCOMES = frozenset({"text", "verified_blank", "explicit_nontext", "accepted_zero_output"})
OUTCOMES = SUCCESS_OUTCOMES | frozenset({"failed", "not_measured", "not_run"})
_NORMALIZATION = "nfkc-collapse-whitespace-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _is_sentinel(value: object) -> bool:
    return isinstance(value, str) and value in SENTINELS


def _measurement(value: object, name: str, *, integer: bool = False) -> None:
    if _is_sentinel(value):
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number or sentinel")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer or sentinel")


def _validate_output(output: object) -> None:
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    required = ("status", "path", "sha256", "bytes", "page_count")
    missing = [field for field in required if field not in output]
    if missing:
        raise ValueError(f"output is missing required fields: {', '.join(missing)}")
    if output["status"] not in {"published", "failed", *SENTINELS}:
        raise ValueError("output.status is invalid")
    if not isinstance(output["path"], str):
        raise ValueError("output.path must be a string or sentinel")
    digest = output["sha256"]
    if not (
        _is_sentinel(digest)
        or isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    ):
        raise ValueError("output.sha256 must be lowercase SHA-256 or sentinel")
    _measurement(output["bytes"], "output.bytes", integer=True)
    _measurement(output["page_count"], "output.page_count", integer=True)


def validate_run_input(run: object) -> dict[str, Any]:
    """Validate a run record without importing or invoking an OCR engine."""
    if not isinstance(run, dict):
        raise ValueError("run must be an object")
    required = ("schema", "run_id", "model_status", "fixtures", "measurements", "output")
    missing = [field for field in required if field not in run]
    if missing:
        raise ValueError(f"run is missing required fields: {', '.join(missing)}")
    if run["schema"] != INPUT_SCHEMA:
        raise ValueError(f"unsupported run schema: {run['schema']!r}")
    if not isinstance(run["run_id"], str) or not run["run_id"].strip():
        raise ValueError("run_id must be a non-empty string")
    if run["model_status"] not in {"not_run", "external", "not_measured"}:
        raise ValueError("model_status is invalid")

    measurements = run["measurements"]
    if not isinstance(measurements, dict):
        raise ValueError("measurements must be an object")
    timing = measurements.get("stage_timing_seconds")
    if not isinstance(timing, dict):
        raise ValueError("measurements.stage_timing_seconds must be an object")
    for stage in STAGES:
        if stage not in timing:
            raise ValueError(f"stage_timing_seconds is missing {stage}")
        _measurement(timing[stage], f"stage_timing_seconds.{stage}")
    for field in ("peak_ram_bytes", "peak_vram_bytes", "engine_invocations", "render_count"):
        if field not in measurements:
            raise ValueError(f"measurements is missing {field}")
        _measurement(measurements[field], f"measurements.{field}", integer=True)

    reuse = measurements.get("chunk_reuse")
    if not isinstance(reuse, dict):
        raise ValueError("measurements.chunk_reuse must be an object")
    for field in ("status", "reused_chunks", "rerun_reasons"):
        if field not in reuse:
            raise ValueError(f"chunk_reuse is missing {field}")
    if reuse["status"] not in {"reused", "not_reused", *SENTINELS}:
        raise ValueError("chunk_reuse.status is invalid")
    _measurement(reuse["reused_chunks"], "chunk_reuse.reused_chunks", integer=True)
    reasons = reuse["rerun_reasons"]
    if not (_is_sentinel(reasons) or isinstance(reasons, list) and all(isinstance(x, str) for x in reasons)):
        raise ValueError("chunk_reuse.rerun_reasons must be a sentinel or string list")

    fixtures = run["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("fixtures must be a non-empty list")

    fixture_ids: set[str] = set()
    for fixture in fixtures:
        if not isinstance(fixture, dict) or "fixture_id" not in fixture or "pages" not in fixture:
            raise ValueError("each fixture requires fixture_id and pages")
        fixture_id = fixture["fixture_id"]
        if not isinstance(fixture_id, str) or not fixture_id.strip() or fixture_id in fixture_ids:
            raise ValueError("fixture_id must be unique and non-empty")
        fixture_ids.add(fixture_id)
        pages = fixture["pages"]
        if not isinstance(pages, list):
            raise ValueError(f"{fixture_id}.pages must be a list")
        for page in pages:
            if not isinstance(page, dict):
                raise ValueError(f"{fixture_id}.pages entries must be objects")
            fields = ("source_page", "output_page", "searchable_text", "outcome")
            missing_page = [field for field in fields if field not in page]
            if missing_page:
                raise ValueError(f"{fixture_id}.page is missing {', '.join(missing_page)}")
            source_page = page["source_page"]
            if isinstance(source_page, bool) or not isinstance(source_page, int) or source_page < 1:
                raise ValueError(f"{fixture_id}.source_page must be a positive integer")
            output_page = page["output_page"]
            if not (
                _is_sentinel(output_page)
                or isinstance(output_page, int)
                and not isinstance(output_page, bool)
                and output_page > 0
            ):
                raise ValueError(f"{fixture_id}.output_page must be positive integer or sentinel")
            searchable_text = page["searchable_text"]
            if not (_is_sentinel(searchable_text) or isinstance(searchable_text, str)):
                raise ValueError(f"{fixture_id}.searchable_text must be text or sentinel")
            if page["outcome"] not in OUTCOMES:
                raise ValueError(f"{fixture_id}.outcome is invalid")
    _validate_output(run["output"])
    return run


def normalize_search_text(text: str) -> str:
    """Normalize searchable text for equality, CER and WER."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _metric(distance: int, reference_units: int, measured_pages: int) -> dict[str, Any]:
    if measured_pages == 0:
        return {
            "status": "not_run",
            "value": "not_run",
            "distance_units": 0,
            "reference_units": 0,
            "measured_pages": 0,
        }
    value = distance / reference_units if reference_units else (0.0 if distance == 0 else 1.0)
    return {
        "status": "measured",
        "value": round(value, 12),
        "distance_units": distance,
        "reference_units": reference_units,
        "measured_pages": measured_pages,
    }


def _load_ground_truth(corpus_root: Path, fixture: dict[str, Any]) -> list[dict[str, Any]]:
    path = corpus_root / str(fixture["ground_truth"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != int(fixture["page_count"]):
        raise ValueError(f"Ground Truth page count mismatch: {fixture['id']}")
    return rows


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}

def _fixture_report(
    fixture: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    expected_by_page = {int(row["page"]): row for row in ground_truth}
    expected_pages = set(expected_by_page)
    rows = list(candidate["pages"])
    rows_by_source: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_source.setdefault(int(row["source_page"]), []).append(row)
    duplicate_source_pages = sorted(
        page for page, page_rows in rows_by_source.items()
        if len(page_rows) > 1 and page in expected_pages
    )
    unexpected_source_pages = sorted(set(rows_by_source) - expected_pages)
    missing_source_pages = sorted(expected_pages - set(rows_by_source))

    char_distance = word_distance = char_reference = word_reference = measured_pages = 0
    exact_pages: list[int] = []
    mismatched_pages: list[int] = []
    incomplete_pages: list[int] = []
    not_run_pages: list[int] = []
    outcome_matches = 0
    observed_outcomes: Counter[str] = Counter(str(row["outcome"]) for row in rows)

    for page_number in sorted(expected_pages):
        page_rows = rows_by_source.get(page_number, [])
        row = page_rows[0] if page_rows else None
        if row is None:
            incomplete_pages.append(page_number)
            not_run_pages.append(page_number)
            continue
        outcome = str(row["outcome"])
        complete = (
            outcome in SUCCESS_OUTCOMES
            and isinstance(row["searchable_text"], str)
            and isinstance(row["output_page"], int)
            and not isinstance(row["output_page"], bool)
        )
        if not complete:
            incomplete_pages.append(page_number)
            if outcome in SENTINELS:
                not_run_pages.append(page_number)
            continue
        if outcome == str(expected_by_page[page_number]["expected_outcome"]):
            outcome_matches += 1
        expected_text = normalize_search_text(str(expected_by_page[page_number]["text"]))
        observed_text = normalize_search_text(str(row["searchable_text"]))
        if expected_text == observed_text:
            exact_pages.append(page_number)
        else:
            mismatched_pages.append(page_number)
        char_distance += _levenshtein(list(expected_text), list(observed_text))
        word_distance += _levenshtein(expected_text.split(), observed_text.split())
        char_reference += len(expected_text)
        word_reference += len(expected_text.split())
        measured_pages += 1

    output_pages = [
        int(row["output_page"])
        for row in rows
        if isinstance(row["output_page"], int) and not isinstance(row["output_page"], bool)
    ]
    output_counter = Counter(output_pages)
    duplicate_output_pages = sorted(page for page, count in output_counter.items() if count > 1)
    wrong_output_pages = sorted(
        int(row["source_page"])
        for row in rows
        if int(row["source_page"]) in expected_pages
        and isinstance(row["output_page"], int)
        and not isinstance(row["output_page"], bool)
        and int(row["output_page"]) != int(row["source_page"])
    )
    expected_output_pages = set(range(1, len(expected_pages) + 1))
    extra_output_pages = sorted(set(output_pages) - expected_output_pages)
    mapping_problems = (
        missing_source_pages or unexpected_source_pages or duplicate_source_pages
        or duplicate_output_pages or wrong_output_pages or extra_output_pages
        or set(output_pages) != expected_output_pages
    )
    mapping_status = "not_run" if not rows else ("fail" if mapping_problems else "pass")
    retention_status = (
        "not_run"
        if measured_pages == 0
        else "pass"
        if len(exact_pages) == len(expected_pages) and not incomplete_pages and not mismatched_pages
        else "fail"
    )
    expected_outcomes = Counter(str(row["expected_outcome"]) for row in ground_truth)
    return {
        "fixture_id": fixture["id"],
        "expected_page_count": len(expected_pages),
        "metrics": {
            "cer": _metric(char_distance, char_reference, measured_pages),
            "wer": _metric(word_distance, word_reference, measured_pages),
        },
        "searchability": {
            "normalization": _NORMALIZATION,
            "exact_text_retention": {
                "status": retention_status,
                "matched_pages": len(exact_pages),
                "measured_pages": measured_pages,
                "mismatched_pages": mismatched_pages,
                "incomplete_pages": sorted(incomplete_pages),
            },
        },
        "page_mapping": {
            "status": mapping_status,
            "expected_page_count": len(expected_pages),
            "observed_source_page_count": len(rows_by_source),
            "mapped_page_count": len(output_pages),
            "missing_source_pages": missing_source_pages,
            "unexpected_source_pages": unexpected_source_pages,
            "duplicate_source_pages": duplicate_source_pages,
            "duplicate_output_pages": duplicate_output_pages,
            "wrong_output_pages": wrong_output_pages,
            "extra_output_pages": extra_output_pages,
        },
        "page_outcomes": {
            "expected": _counter_dict(expected_outcomes),
            "observed": _counter_dict(observed_outcomes),
            "matched_expected_pages": outcome_matches,
            "not_run_pages": sorted(not_run_pages),
        },
    }

def _aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    char_distance = char_reference = word_distance = word_reference = measured_pages = 0
    expected_pages = matched_pages = measured_retention_pages = 0
    incomplete_pages = mismatched_pages = 0
    expected_outcomes: Counter[str] = Counter()
    observed_outcomes: Counter[str] = Counter()
    mapping_statuses: list[str] = []
    for report in reports:
        cer = report["metrics"]["cer"]
        wer = report["metrics"]["wer"]
        if cer["status"] == "measured":
            char_distance += int(cer["distance_units"])
            char_reference += int(cer["reference_units"])
            measured_pages += int(cer["measured_pages"])
        if wer["status"] == "measured":
            word_distance += int(wer["distance_units"])
            word_reference += int(wer["reference_units"])
        expected_pages += int(report["expected_page_count"])
        retention = report["searchability"]["exact_text_retention"]
        matched_pages += int(retention["matched_pages"])
        measured_retention_pages += int(retention["measured_pages"])
        incomplete_pages += len(retention["incomplete_pages"])
        mismatched_pages += len(retention["mismatched_pages"])
        expected_outcomes.update(report["page_outcomes"]["expected"])
        observed_outcomes.update(report["page_outcomes"]["observed"])
        mapping_statuses.append(str(report["page_mapping"]["status"]))
    retention_status = (
        "not_run"
        if measured_retention_pages == 0
        else "pass"
        if matched_pages == expected_pages and incomplete_pages == 0 and mismatched_pages == 0
        else "fail"
    )
    if not mapping_statuses or all(status == "not_run" for status in mapping_statuses):
        mapping_status = "not_run"
    elif all(status == "pass" for status in mapping_statuses):
        mapping_status = "pass"
    else:
        mapping_status = "fail"
    return {
        "metrics": {
            "cer": _metric(char_distance, char_reference, measured_pages),
            "wer": _metric(word_distance, word_reference, measured_pages),
        },
        "searchability": {
            "normalization": _NORMALIZATION,
            "exact_text_retention": {
                "status": retention_status,
                "matched_pages": matched_pages,
                "measured_pages": measured_retention_pages,
                "expected_pages": expected_pages,
                "mismatched_pages": mismatched_pages,
                "incomplete_pages": incomplete_pages,
            },
        },
        "page_mapping": {"status": mapping_status, "fixture_count": len(reports)},
        "page_outcomes": {
            "expected": _counter_dict(expected_outcomes),
            "observed": _counter_dict(observed_outcomes),
        },
    }


def evaluate_run(corpus_root: Path, run: object) -> dict[str, Any]:
    """Evaluate supplied searchable text and metadata; never invoke OCR or models."""
    manifest = validate_corpus(corpus_root)
    validated = validate_run_input(run)
    fixture_by_id = {fixture["id"]: fixture for fixture in manifest["fixtures"]}
    reports: list[dict[str, Any]] = []
    for candidate in sorted(validated["fixtures"], key=lambda item: str(item["fixture_id"])):
        fixture_id = str(candidate["fixture_id"])
        if fixture_id not in fixture_by_id:
            raise ValueError(f"Unknown fixture: {fixture_id}")
        reports.append(
            _fixture_report(
                fixture_by_id[fixture_id],
                _load_ground_truth(corpus_root, fixture_by_id[fixture_id]),
                candidate,
            )
        )
    return {
        "schema": REPORT_SCHEMA,
        "run_id": validated["run_id"],
        "model_status": validated["model_status"],
        "evaluator": {
            "name": "model-free-synthetic-evaluator",
            "version": EVALUATOR_VERSION,
            "model_invoked": False,
            "normalization": _NORMALIZATION,
            "output_metadata": "externally_attested_not_verified",
        },
        "measurements": copy.deepcopy(validated["measurements"]),
        "output": copy.deepcopy(validated["output"]),
        "fixtures": reports,
        "summary": _aggregate(reports),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="JSON evaluator input.")
    parser.add_argument("--output", required=True, type=Path, help="JSON report destination.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run = json.loads(args.run.read_text(encoding="utf-8"))
    report = evaluate_run(args.corpus, run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_canonical_json(report), encoding="utf-8", newline="\n")
    print(f"evaluation report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
