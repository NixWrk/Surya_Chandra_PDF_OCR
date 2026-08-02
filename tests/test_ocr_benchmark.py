from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

import uniscan.ocr.benchmark as ocr_benchmark_mod
from uniscan.cli import main
from uniscan.export import export_pages_as_pdf
from uniscan.ocr import (
    OCR_ENGINE_CHANDRA,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_OLMOCR,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_PYMUPDF,
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_SURYA,
    resolve_pdf_page_indices,
    run_ocr_benchmark,
    sample_pdf_page_indices,
)

FIXTURE_PDF = Path(r"J:\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf")
ALL_ENGINES = (
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PYMUPDF,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_SURYA,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_CHANDRA,
    OCR_ENGINE_OLMOCR,
)
SEARCHABLE_ENGINES = (
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PYMUPDF,
)
EXTRACTION_ENGINES = (
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_SURYA,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_CHANDRA,
    OCR_ENGINE_OLMOCR,
)


def _write_fixture_png(path: Path) -> None:
    Image.new("RGB", (120, 80), color=(96, 96, 96)).save(path, format="PNG")


def _load_fixture_chandra_image(path: str, min_image_dim: int = 1536) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.width < min_image_dim or image.height < min_image_dim:
        scale = min_image_dim / float(min(image.width, image.height))
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return image


def test_collect_text_strings_rejects_invalid_utf8_bytes() -> None:
    assert ocr_benchmark_mod._collect_text_strings(b"valid") == ["valid"]
    assert ocr_benchmark_mod._collect_text_strings(b"invalid:\xff") == []


def _build_sample_pdf(tmp_path: Path, page_values: list[int]) -> Path:
    pages: list[np.ndarray] = []
    for value in page_values:
        pages.append(np.full((120, 180, 3), value, dtype=np.uint8))
    pdf_path = tmp_path / "fixture.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=150)
    return pdf_path


def _ready_status(engine_name: str, *, searchable_pdf: bool) -> SimpleNamespace:
    return SimpleNamespace(
        engine_name=engine_name,
        ready=True,
        missing=[],
        searchable_pdf=searchable_pdf,
        label=engine_name,
    )


def test_sample_pdf_page_indices_returns_evenly_distributed_pages() -> None:
    assert sample_pdf_page_indices(12, sample_size=3) == [0, 6, 11]
    assert sample_pdf_page_indices(12, sample_size=2) == [0, 11]
    assert sample_pdf_page_indices(12, sample_size=1) == [0]
    assert sample_pdf_page_indices(2, sample_size=5) == [0, 1]
    assert sample_pdf_page_indices(0, sample_size=5) == []


def test_resolve_pdf_page_indices_with_explicit_pages() -> None:
    assert resolve_pdf_page_indices(12, page_numbers=[3, 9, 3]) == [2, 8]
    assert resolve_pdf_page_indices(12, sample_size=2) == [0, 11]

    with pytest.raises(ValueError, match=">= 1"):
        resolve_pdf_page_indices(12, page_numbers=[0])
    with pytest.raises(ValueError, match="valid range is 1..12"):
        resolve_pdf_page_indices(12, page_numbers=[13])


def test_render_sample_paths_streams_ten_page_pdf(tmp_path) -> None:
    pdf_path = _build_sample_pdf(tmp_path, list(range(10, 110, 10)))
    tmp_dir = tmp_path / "rendered"
    tmp_dir.mkdir()

    paths = ocr_benchmark_mod._render_sample_paths(
        pdf_path,
        list(range(10)),
        dpi=72,
        tmp_dir=tmp_dir,
    )

    assert [path.name for path in paths] == [f"{idx:05d}.png" for idx in range(1, 11)]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_safe_render_dpi_caps_extreme_page_memory() -> None:
    from uniscan.io.loaders import _MAX_RENDER_PIXELS, _safe_render_dpi

    page_rect = SimpleNamespace(width=2384.0, height=3370.0)
    with pytest.warns(UserWarning, match="render memory limit"):
        safe_dpi = _safe_render_dpi(page_rect, 220)

    assert 1 <= safe_dpi < 220
    rendered_pixels = page_rect.width / 72.0 * safe_dpi * (page_rect.height / 72.0 * safe_dpi)
    assert rendered_pixels <= _MAX_RENDER_PIXELS


def test_run_ocr_benchmark_writes_report_and_artifacts(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [30, 90, 150])
    output_dir = tmp_path / "out"

    def fake_status(engine_name: str, **_kwargs):
        return _ready_status(engine_name, searchable_pdf=engine_name in SEARCHABLE_ENGINES)

    def fake_searchable_pdf(image_paths, *, out_pdf, lang, engine_name):
        out_pdf.write_text(f"{engine_name}:{lang}:{len(image_paths)}", encoding="utf-8")
        return out_pdf

    def fake_extract_pdf_text(_pdf_path: Path) -> str:
        return "x" * 321

    def fake_paddleocr(image_paths, *, lang):
        assert len(image_paths) == 1
        return f"paddle:{lang}:{len(image_paths)}", 12

    def fake_surya(image_paths, *, lang, work_dir, which_fn, run_cmd):
        assert "surya_work" in str(work_dir)
        assert len(image_paths) == 2
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "image_name": f"{idx:04d}_{image_path.name}",
                            "pages": [
                                {
                                    "image_bbox": [0, 0, 1, 1],
                                    "text_lines": [{"text": "x" * 13, "bbox": [0, 0, 1, 1]}],
                                }
                            ],
                        }
                        for idx, image_path in enumerate(image_paths, start=1)
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return f"surya:{lang}:{len(image_paths)}", 26

    def fake_mineru(image_paths, *, lang, work_dir, which_fn, run_cmd):
        assert "mineru_work" in str(work_dir)
        assert len(image_paths) == 1
        return f"mineru:{lang}:{len(image_paths)}", 14

    def fake_chandra(
        image_paths,
        *,
        lang,
        work_dir,
        which_fn,
        run_cmd,
        page_progress_cb=None,
        source_raster_identities=None,
    ):
        assert source_raster_identities is not None
        assert len(source_raster_identities) == len(image_paths)
        assert "chandra_work" in str(work_dir)
        assert len(image_paths) == 2
        if page_progress_cb is not None:
            page_progress_cb(1, 2)
            page_progress_cb(2, 2)
        sidecar = work_dir / "chandra_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar_payload = {
            "images": [
                {
                    "image_name": image_paths[0].name,
                    "pages": [{"text_lines": [{"text": "x" * 15, "bbox": [0, 0, 1, 1]}]}],
                },
                {
                    "image_name": image_paths[1].name,
                    "pages": [{"text_lines": [{"text": "x" * 15, "bbox": [0, 0, 1, 1]}]}],
                },
            ]
        }
        sidecar.write_text(json.dumps(sidecar_payload, ensure_ascii=False), encoding="utf-8")
        return f"chandra:{lang}:{len(image_paths)}", 30

    def fake_olmocr(image_paths, *, lang, work_dir, which_fn, run_cmd):
        assert "olmocr_work" in str(work_dir)
        assert len(image_paths) == 1
        return f"olmocr:{lang}:{len(image_paths)}", 16

    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.benchmark.image_paths_to_searchable_pdf", fake_searchable_pdf)
    monkeypatch.setattr("uniscan.ocr.benchmark._extract_pdf_text", fake_extract_pdf_text)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_paddleocr_direct", fake_paddleocr)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_surya_direct", fake_surya)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_mineru_direct", fake_mineru)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_chandra_direct", fake_chandra)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_olmocr_direct", fake_olmocr)

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=ALL_ENGINES,
        sample_size=2,
        dpi=120,
        lang="eng",
    )

    assert [result.engine for result in results] == list(ALL_ENGINES)
    assert all(result.status == "ok" for result in results)
    assert all(result.artifact_path and Path(result.artifact_path).exists() for result in results)
    for result in results:
        assert result.sample_pages == [1, 3]
    assert {result.text_chars for result in results if result.engine in SEARCHABLE_ENGINES} == {321}
    assert {result.text_chars for result in results if result.engine in EXTRACTION_ENGINES} == {
        24,
        26,
        28,
        30,
        32,
    }

    # Extraction engines now persist page-aware artifacts and markerized aggregate.
    for engine in EXTRACTION_ENGINES:
        engine_dir = output_dir / engine
        assert (engine_dir / "pages.json").exists()
        assert (engine_dir / "all_pages.txt").exists()
        assert (engine_dir / "page_0001.txt").exists()
        assert (engine_dir / "page_0003.txt").exists()

    report_path = output_dir / "fixture_ocr_benchmark.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["pdf_path"] == str(pdf_path)
    assert payload["sample_pages"] == [1, 3]
    assert len(payload["results"]) == len(ALL_ENGINES)


def test_run_ocr_benchmark_uses_explicit_page_numbers(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [30, 90, 150])
    output_dir = tmp_path / "out"

    def fake_sample(_page_count: int, *, sample_size: int = 5) -> list[int]:
        raise AssertionError(
            f"sample_pdf_page_indices should not be called, sample_size={sample_size}"
        )

    def fake_status(engine_name: str, **_kwargs):
        return _ready_status(engine_name, searchable_pdf=False)

    def fake_paddleocr(image_paths, *, lang):
        return f"paddle:{lang}:{len(image_paths)}", 7

    monkeypatch.setattr("uniscan.ocr.benchmark.sample_pdf_page_indices", fake_sample)
    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_paddleocr_direct", fake_paddleocr)

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PADDLEOCR,),
        sample_size=99,
        page_numbers=(2,),
        dpi=120,
        lang="eng",
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].sample_pages == [2]
    report = json.loads((output_dir / "fixture_ocr_benchmark.json").read_text(encoding="utf-8"))
    assert report["sample_pages"] == [2]


def test_run_ocr_benchmark_unready_engine_is_error(tmp_path, monkeypatch) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [30, 90, 150])
    output_dir = tmp_path / "out"

    def fake_status(engine_name: str, **_kwargs):
        return SimpleNamespace(
            engine_name=engine_name,
            ready=False,
            missing=["dependency-x"],
            searchable_pdf=False,
            label=engine_name,
        )

    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_SURYA,),
        sample_size=1,
        dpi=100,
    )

    assert len(results) == 1
    assert results[0].engine == OCR_ENGINE_SURYA
    assert results[0].status == "error"
    assert results[0].note == "missing: dependency-x"
    assert results[0].artifact_path is not None


def test_run_ocr_benchmark_seals_surya_retry_failure_before_runtime_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [90])
    output_dir = tmp_path / "out"
    runtime_dir = tmp_path / "runtime"
    expected_error = "Invalid third-attempt geometry: line text has no alphanumeric evidence."

    def allocate_runtime(*, prefix: str) -> Path:
        assert prefix == "uniscan_ocr_benchmark_"
        runtime_dir.mkdir()
        return runtime_dir

    def fail_pagewise(*_args, work_dir: Path, **_kwargs):
        attempt_root = work_dir / "zero_output_retry" / "page_0001" / "attempt_3_scaled"
        input_path = attempt_root / "input" / "00001.png"
        raw_path = attempt_root / "module" / "surya_out" / "surya_input" / "results.json"
        sidecar_path = attempt_root / "module" / "surya_page_lines.json"
        input_path.parent.mkdir(parents=True)
        raw_path.parent.mkdir(parents=True)
        input_path.write_bytes(b"sealed retry image")
        raw_path.write_text('{"00001":[{"text_lines":[{"text":"---"}]}]}', encoding="utf-8")
        sidecar_path.write_text(
            '{"execution_path":"module","images":[{"image_name":"00001.png"}]}',
            encoding="utf-8",
        )
        raise RuntimeError(expected_error)

    monkeypatch.setattr(ocr_benchmark_mod, "_create_runtime_work_dir", allocate_runtime)
    monkeypatch.setattr(
        ocr_benchmark_mod,
        "detect_ocr_engine_status",
        lambda engine_name, **_kwargs: _ready_status(engine_name, searchable_pdf=False),
    )
    monkeypatch.setattr(ocr_benchmark_mod, "_run_extraction_engine_pagewise", fail_pagewise)

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_SURYA,),
        sample_size=1,
        dpi=100,
    )

    assert results[0].status == "error"
    assert results[0].error == expected_error
    assert not runtime_dir.exists()
    report = json.loads((output_dir / "fixture_ocr_benchmark.json").read_text(encoding="utf-8"))
    diagnostics = report["failure_diagnostics"]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["status"] == "saved"
    assert diagnostic["original_error"] == expected_error
    evidence_root = Path(diagnostic["evidence_root"])
    manifest_path = Path(diagnostic["manifest_path"])
    assert manifest_path == evidence_root / "manifest.json"
    assert ocr_benchmark_mod._sha256_path(manifest_path) == diagnostic["manifest_sha256"]
    assert manifest_path.stat().st_size == diagnostic["manifest_bytes"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "uniscan.surya-failure-evidence.v1"
    assert manifest["status"] == "sealed"
    assert manifest["original_error"] == expected_error
    expected_paths = {
        "payload/zero_output_retry/page_0001/attempt_3_scaled/input/00001.png",
        "payload/zero_output_retry/page_0001/attempt_3_scaled/module/surya_out/surya_input/results.json",
        "payload/zero_output_retry/page_0001/attempt_3_scaled/module/surya_page_lines.json",
    }
    assert {item["path"] for item in manifest["files"]} == expected_paths
    assert diagnostic["file_count"] == len(expected_paths)
    assert diagnostic["total_bytes"] == manifest["total_bytes"]
    assert manifest["file_count"] == len(expected_paths)
    assert manifest["limits"] == {
        "max_files": ocr_benchmark_mod._MAX_SURYA_FAILURE_EVIDENCE_FILES,
        "max_entries": ocr_benchmark_mod._MAX_SURYA_FAILURE_EVIDENCE_ENTRIES,
        "max_relative_path_chars": (
            ocr_benchmark_mod._MAX_SURYA_FAILURE_EVIDENCE_RELATIVE_PATH_CHARS
        ),
        "max_bytes": ocr_benchmark_mod._MAX_SURYA_FAILURE_EVIDENCE_BYTES,
    }
    assert manifest["payload_root"] == "payload"
    for item in manifest["files"]:
        evidence_path = evidence_root / item["path"]
        assert evidence_path.stat().st_size == item["bytes"]
        assert ocr_benchmark_mod._sha256_path(evidence_path) == item["sha256"]


def _failure_snapshot_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    trusted_root = tmp_path / "runtime"
    source_root = trusted_root / "surya_work"
    output_dir = tmp_path / "out"
    trusted_root.mkdir()
    output_dir.mkdir()
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"sealed pdf")
    return trusted_root, source_root, output_dir, pdf_path


def test_surya_failure_snapshot_missing_source_returns_none(tmp_path: Path) -> None:
    trusted_root, source_root, output_dir, pdf_path = _failure_snapshot_fixture(tmp_path)

    diagnostic = ocr_benchmark_mod._snapshot_surya_failure_evidence(
        source_root=source_root,
        trusted_root=trusted_root,
        output_dir=output_dir,
        pdf_path=pdf_path,
        sample_pages_1based=[1],
        dpi=300,
        lang="eng",
        error="failure before Surya work materialized",
    )

    assert diagnostic is None
    assert list(output_dir.iterdir()) == []


def test_surya_failure_snapshot_seals_pdf_symlink_target_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_root, source_root, output_dir, _pdf_path = _failure_snapshot_fixture(tmp_path)
    source_root.mkdir()
    (source_root / "raw.json").write_text("{}", encoding="utf-8")
    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"
    first_target.write_bytes(b"first sealed PDF")
    second_target.write_bytes(b"second PDF")
    linked_pdf = tmp_path / "linked.pdf"
    try:
        linked_pdf.symlink_to(first_target)
    except OSError:
        pytest.skip("file symlinks are unavailable on this host")
    real_fingerprint = ocr_benchmark_mod._stable_file_fingerprint
    retargeted = False

    def retarget_after_first_pdf_seal(path: Path) -> dict[str, object]:
        nonlocal retargeted
        fingerprint = real_fingerprint(path)
        if Path(path) == first_target and not retargeted:
            linked_pdf.unlink()
            linked_pdf.symlink_to(second_target)
            retargeted = True
        return fingerprint

    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_stable_file_fingerprint",
        retarget_after_first_pdf_seal,
    )
    diagnostic = ocr_benchmark_mod._snapshot_surya_failure_evidence(
        source_root=source_root,
        trusted_root=trusted_root,
        output_dir=output_dir,
        pdf_path=linked_pdf,
        sample_pages_1based=[1],
        dpi=300,
        lang="eng",
        error="boom",
    )

    assert retargeted is True
    assert diagnostic is not None
    manifest = json.loads(Path(diagnostic["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["source_pdf"] == {
        "path": str(first_target.resolve()),
        "sha256": ocr_benchmark_mod._sha256_path(first_target),
        "bytes": first_target.stat().st_size,
    }


def test_surya_failure_snapshot_rejects_linked_source_root(tmp_path: Path) -> None:
    trusted_root, source_root, output_dir, pdf_path = _failure_snapshot_fixture(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.txt").write_text("outside", encoding="utf-8")
    try:
        source_root.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")
    with pytest.raises(RuntimeError, match="source root is unsafe"):
        ocr_benchmark_mod._snapshot_surya_failure_evidence(
            source_root=source_root,
            trusted_root=trusted_root,
            output_dir=output_dir,
            pdf_path=pdf_path,
            sample_pages_1based=[1],
            dpi=300,
            lang="eng",
            error="boom",
        )
    source_root.unlink()
    assert list(output_dir.iterdir()) == []


def test_surya_failure_snapshot_rejects_reparse_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_root, source_root, output_dir, pdf_path = _failure_snapshot_fixture(tmp_path)
    source_root.mkdir()
    source_stat = source_root.lstat()
    source_identity = (source_stat.st_dev, source_stat.st_ino)
    real_is_reparse = ocr_benchmark_mod._stat_has_reparse_point
    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_stat_has_reparse_point",
        lambda path_stat: (
            (path_stat.st_dev, path_stat.st_ino) == source_identity or real_is_reparse(path_stat)
        ),
    )
    with pytest.raises(RuntimeError, match="source root is unsafe"):
        ocr_benchmark_mod._snapshot_surya_failure_evidence(
            source_root=source_root,
            trusted_root=trusted_root,
            output_dir=output_dir,
            pdf_path=pdf_path,
            sample_pages_1based=[1],
            dpi=300,
            lang="eng",
            error="boom",
        )
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "files", "message"),
    [
        ("_MAX_SURYA_FAILURE_EVIDENCE_FILES", 1, (b"a", b"b"), "file-count limit"),
        ("_MAX_SURYA_FAILURE_EVIDENCE_ENTRIES", 1, (b"a", b"b"), "entry-count limit"),
        (
            "_MAX_SURYA_FAILURE_EVIDENCE_RELATIVE_PATH_CHARS",
            3,
            (b"a",),
            "relative-path-length limit",
        ),
        ("_MAX_SURYA_FAILURE_EVIDENCE_BYTES", 3, (b"four",), "byte limit"),
    ],
)
def test_surya_failure_snapshot_is_bounded_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    files: tuple[bytes, ...],
    message: str,
) -> None:
    trusted_root, source_root, output_dir, pdf_path = _failure_snapshot_fixture(tmp_path)
    source_root.mkdir()
    for index, payload in enumerate(files):
        (source_root / f"{index}.bin").write_bytes(payload)
    monkeypatch.setattr(ocr_benchmark_mod, limit_name, limit_value)

    with pytest.raises(RuntimeError, match=message):
        ocr_benchmark_mod._snapshot_surya_failure_evidence(
            source_root=source_root,
            trusted_root=trusted_root,
            output_dir=output_dir,
            pdf_path=pdf_path,
            sample_pages_1based=[1],
            dpi=300,
            lang="eng",
            error="boom",
        )

    assert list(output_dir.iterdir()) == []


def test_surya_failure_snapshot_isolates_source_manifest_namespace(tmp_path: Path) -> None:
    trusted_root, source_root, output_dir, pdf_path = _failure_snapshot_fixture(tmp_path)
    source_root.mkdir()
    source_manifest = source_root / "manifest.json"
    source_manifest.write_text("source-owned manifest", encoding="utf-8")

    diagnostic = ocr_benchmark_mod._snapshot_surya_failure_evidence(
        source_root=source_root,
        trusted_root=trusted_root,
        output_dir=output_dir,
        pdf_path=pdf_path,
        sample_pages_1based=[1],
        dpi=300,
        lang="eng",
        error="boom",
    )

    assert diagnostic is not None
    evidence_root = Path(diagnostic["evidence_root"])
    assert (evidence_root / "manifest.json").is_file()
    assert (evidence_root / "payload" / "manifest.json").read_text(encoding="utf-8") == (
        "source-owned manifest"
    )
    manifest = json.loads((evidence_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == [
        {
            "path": "payload/manifest.json",
            "sha256": ocr_benchmark_mod._sha256_path(evidence_root / "payload" / "manifest.json"),
            "bytes": len("source-owned manifest"),
        }
    ]


def test_surya_failure_snapshot_rechecks_trusted_root_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_root, source_root, output_dir, pdf_path = _failure_snapshot_fixture(tmp_path)
    source_root.mkdir()
    (source_root / "raw.json").write_text("{}", encoding="utf-8")
    real_check = ocr_benchmark_mod._strict_snapshot_directory
    rechecks = 0

    def inject_identity_change(
        path: Path,
        *,
        label: str,
        expected: os.stat_result | None = None,
    ) -> os.stat_result:
        nonlocal rechecks
        if Path(path) == trusted_root and expected is not None:
            rechecks += 1
            raise RuntimeError("Surya failure evidence trusted runtime root changed while copying")
        return real_check(path, label=label, expected=expected)

    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_strict_snapshot_directory",
        inject_identity_change,
    )
    with pytest.raises(RuntimeError, match="trusted runtime root changed"):
        ocr_benchmark_mod._snapshot_surya_failure_evidence(
            source_root=source_root,
            trusted_root=trusted_root,
            output_dir=output_dir,
            pdf_path=pdf_path,
            sample_pages_1based=[1],
            dpi=300,
            lang="eng",
            error="boom",
        )
    assert rechecks == 1
    assert list(output_dir.iterdir()) == []


def test_surya_failure_snapshot_removes_published_orphan_on_descriptor_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_root, source_root, output_dir, pdf_path = _failure_snapshot_fixture(tmp_path)
    source_root.mkdir()
    (source_root / "raw.json").write_text("{}", encoding="utf-8")
    real_fingerprint = ocr_benchmark_mod._stable_file_fingerprint

    def fail_manifest_descriptor(path: Path) -> dict[str, object]:
        if path.name == "manifest.json":
            raise OSError("injected descriptor failure")
        return real_fingerprint(path)

    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_stable_file_fingerprint",
        fail_manifest_descriptor,
    )
    with pytest.raises(OSError, match="injected descriptor failure"):
        ocr_benchmark_mod._snapshot_surya_failure_evidence(
            source_root=source_root,
            trusted_root=trusted_root,
            output_dir=output_dir,
            pdf_path=pdf_path,
            sample_pages_1based=[1],
            dpi=300,
            lang="eng",
            error="boom",
        )
    assert list(output_dir.iterdir()) == []


def test_bounded_snapshot_copy_rejects_short_eof_without_stat_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"sealed bytes")
    source_stat = source.lstat()
    real_path_open = Path.open

    class ShortReadStream:
        def __init__(self, stream) -> None:
            self._stream = stream
            self._read_once = False

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            self._stream.close()

        def fileno(self) -> int:
            return self._stream.fileno()

        def read(self, _size: int = -1) -> bytes:
            if self._read_once:
                return b""
            self._read_once = True
            return self._stream.read(1)

    def short_source_open(path: Path, *args, **kwargs):
        stream = real_path_open(path, *args, **kwargs)
        if path == source and args and args[0] == "rb":
            return ShortReadStream(stream)
        return stream

    monkeypatch.setattr(Path, "open", short_source_open)
    with pytest.raises(RuntimeError, match="ended before its sealed size"):
        ocr_benchmark_mod._copy_bounded_snapshot_file(
            source=source,
            source_stat=source_stat,
            destination=destination,
            max_bytes=source_stat.st_size,
        )
    assert not destination.exists()


def test_surya_failure_snapshot_error_never_masks_original_ocr_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [90])
    output_dir = tmp_path / "out"
    runtime_dir = tmp_path / "runtime"
    expected_error = "original OCR failure"

    def allocate_runtime(*, prefix: str) -> Path:
        assert prefix == "uniscan_ocr_benchmark_"
        runtime_dir.mkdir()
        return runtime_dir

    def fail_with_hardlink(*_args, work_dir: Path, **_kwargs):
        retry_root = work_dir / "zero_output_retry" / "page_0001"
        retry_root.mkdir(parents=True)
        original = retry_root / "original.json"
        alias = retry_root / "alias.json"
        original.write_text("{}", encoding="utf-8")
        os.link(original, alias)
        raise RuntimeError(expected_error)

    monkeypatch.setattr(ocr_benchmark_mod, "_create_runtime_work_dir", allocate_runtime)
    monkeypatch.setattr(
        ocr_benchmark_mod,
        "detect_ocr_engine_status",
        lambda engine_name, **_kwargs: _ready_status(engine_name, searchable_pdf=False),
    )
    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_run_extraction_engine_pagewise",
        fail_with_hardlink,
    )

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_SURYA,),
        sample_size=1,
        dpi=100,
    )

    assert results[0].status == "error"
    assert results[0].error == expected_error
    assert not runtime_dir.exists()
    report = json.loads((output_dir / "fixture_ocr_benchmark.json").read_text(encoding="utf-8"))
    diagnostics = report["failure_diagnostics"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["status"] == "error"
    assert diagnostics[0]["original_error"] == expected_error
    assert "exclusively owned file" in diagnostics[0]["snapshot_error"]
    assert list(output_dir.glob("fixture_surya_failure_evidence_*")) == []


def test_run_ocr_benchmark_rejects_unsafe_engine_before_touching_output(tmp_path) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="Unsupported OCR engine"):
        run_ocr_benchmark(
            pdf_path=tmp_path / "missing.pdf",
            output_dir=output_dir,
            engines=("../../outside",),
        )

    assert not output_dir.exists()
    assert not (tmp_path / "outside").exists()


def test_collect_olmocr_prefers_markdown_without_duplicate_json_text(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    markdown_dir = workspace / "markdown"
    markdown_dir.mkdir(parents=True)
    (markdown_dir / "document.md").write_text("PRIMARY TEXT", encoding="utf-8")
    (workspace / "document.json").write_text(
        json.dumps({"text": "DUPLICATE TEXT"}),
        encoding="utf-8",
    )

    text, chars = ocr_benchmark_mod._collect_olmocr_workspace_text(workspace)

    assert text == "PRIMARY TEXT"
    assert chars == len(text)


def test_collect_olmocr_rejects_invalid_utf8_and_uses_valid_fallback(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    markdown_dir = workspace / "markdown"
    markdown_dir.mkdir(parents=True)
    (markdown_dir / "broken.md").write_bytes(b"corrupt\xfftext")
    (workspace / "document.json").write_text(
        json.dumps({"text": "VALID FALLBACK"}),
        encoding="utf-8",
    )

    text, chars = ocr_benchmark_mod._collect_olmocr_workspace_text(workspace)

    assert text == "VALID FALLBACK"
    assert chars == len(text)


def test_read_utf8_artifact_is_bounded(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "large.txt"
    artifact.write_bytes(b"12345")
    monkeypatch.setattr(ocr_benchmark_mod, "_MAX_OCR_TEXT_ARTIFACT_BYTES", 4)

    with pytest.raises(RuntimeError, match="exceeds 4 bytes"):
        ocr_benchmark_mod._read_utf8_artifact(artifact)


def test_olmocr_docker_defaults_single_page_to_permissive_error_rate(tmp_path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    image_paths = [tmp_path / "p1.png"]
    image_paths[0].write_bytes(b"fake")
    captured: dict[str, list[str]] = {}

    monkeypatch.delenv("UNISCAN_OLMOCR_DOCKER_MAX_PAGE_ERROR_RATE", raising=False)
    monkeypatch.delenv("UNISCAN_OLMOCR_DOCKER_MAX_PAGE_RETRIES", raising=False)
    monkeypatch.delenv("UNISCAN_OLMOCR_DOCKER_PAGES_PER_GROUP", raising=False)

    def fake_render(_image_paths, out_pdf):
        out_pdf.write_bytes(b"%PDF-1.4\n")

    def fake_collect(_workspace: Path):
        return "ok", 2

    def fake_run(command, capture_output, text):
        captured["command"] = command
        workspace_dir = work_dir / "olmocr_docker" / "work" / "ws"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ocr_benchmark_mod, "_render_images_to_pdf", fake_render)
    monkeypatch.setattr(ocr_benchmark_mod, "_collect_olmocr_workspace_text", fake_collect)

    text, chars = ocr_benchmark_mod._run_olmocr_docker(
        image_paths,
        work_dir=work_dir,
        which_fn=lambda _name: "docker",
        run_cmd=fake_run,
    )

    assert text == "ok"
    assert chars == 2
    command = captured["command"]
    assert "--max_page_error_rate" in command
    assert command[command.index("--max_page_error_rate") + 1] == "1.0"


def test_olmocr_docker_defaults_multi_page_to_relaxed_error_rate(tmp_path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    image_paths = [tmp_path / "p1.png", tmp_path / "p2.png"]
    image_paths[0].write_bytes(b"fake")
    image_paths[1].write_bytes(b"fake")
    captured: dict[str, list[str]] = {}

    monkeypatch.delenv("UNISCAN_OLMOCR_DOCKER_MAX_PAGE_ERROR_RATE", raising=False)

    def fake_render(_image_paths, out_pdf):
        out_pdf.write_bytes(b"%PDF-1.4\n")

    def fake_collect(_workspace: Path):
        return "ok", 2

    def fake_run(command, capture_output, text):
        captured["command"] = command
        workspace_dir = work_dir / "olmocr_docker" / "work" / "ws"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ocr_benchmark_mod, "_render_images_to_pdf", fake_render)
    monkeypatch.setattr(ocr_benchmark_mod, "_collect_olmocr_workspace_text", fake_collect)

    text, chars = ocr_benchmark_mod._run_olmocr_docker(
        image_paths,
        work_dir=work_dir,
        which_fn=lambda _name: "docker",
        run_cmd=fake_run,
    )

    assert text == "ok"
    assert chars == 2
    command = captured["command"]
    assert command[command.index("--max_page_error_rate") + 1] == "0.10"


def test_olmocr_docker_respects_error_rate_overrides(tmp_path, monkeypatch) -> None:
    work_dir = tmp_path / "work"
    image_paths = [tmp_path / "p1.png"]
    image_paths[0].write_bytes(b"fake")
    captured: dict[str, list[str]] = {}

    monkeypatch.setenv("UNISCAN_OLMOCR_DOCKER_PAGES_PER_GROUP", "6")
    monkeypatch.setenv("UNISCAN_OLMOCR_DOCKER_MAX_PAGE_RETRIES", "12")
    monkeypatch.setenv("UNISCAN_OLMOCR_DOCKER_MAX_PAGE_ERROR_RATE", "0.25")

    def fake_render(_image_paths, out_pdf):
        out_pdf.write_bytes(b"%PDF-1.4\n")

    def fake_collect(_workspace: Path):
        return "ok", 2

    def fake_run(command, capture_output, text):
        captured["command"] = command
        workspace_dir = work_dir / "olmocr_docker" / "work" / "ws"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ocr_benchmark_mod, "_render_images_to_pdf", fake_render)
    monkeypatch.setattr(ocr_benchmark_mod, "_collect_olmocr_workspace_text", fake_collect)

    text, chars = ocr_benchmark_mod._run_olmocr_docker(
        image_paths,
        work_dir=work_dir,
        which_fn=lambda _name: "docker",
        run_cmd=fake_run,
    )

    assert text == "ok"
    assert chars == 2
    command = captured["command"]
    assert command[command.index("--pages_per_group") + 1] == "6"
    assert command[command.index("--max_page_retries") + 1] == "12"
    assert command[command.index("--max_page_error_rate") + 1] == "0.25"


def test_run_extraction_engine_pagewise_keeps_partial_results(tmp_path, monkeypatch) -> None:
    image_paths = [tmp_path / "p1.png", tmp_path / "p2.png"]
    for image_path in image_paths:
        _write_fixture_png(image_path)

    def fake_extract(engine, image_paths, *, lang, work_dir, which_fn, run_cmd):
        assert engine == OCR_ENGINE_OLMOCR
        assert len(image_paths) == 1
        if image_paths[0].name == "p2.png":
            raise RuntimeError("page failed")
        return "ok-page", 7

    monkeypatch.setattr(ocr_benchmark_mod, "_run_extraction_engine", fake_extract)

    page_texts, chars, page_errors, page_metadata = (
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_OLMOCR,
            image_paths,
            source_pages_1based=[1, 2],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )
    )

    assert page_texts == ["ok-page", ""]
    assert chars == 7
    assert len(page_errors) == 1
    assert page_errors[0]["source_page"] == 2
    assert page_metadata == []


def test_run_extraction_engine_pagewise_raises_when_all_pages_fail(tmp_path, monkeypatch) -> None:
    image_paths = [tmp_path / "p1.png", tmp_path / "p2.png"]
    for image_path in image_paths:
        _write_fixture_png(image_path)

    def fake_extract(_engine, _image_paths, *, lang, work_dir, which_fn, run_cmd):
        raise RuntimeError("all failed")

    monkeypatch.setattr(ocr_benchmark_mod, "_run_extraction_engine", fake_extract)

    with pytest.raises(RuntimeError, match="all sampled pages failed"):
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_OLMOCR,
            image_paths,
            source_pages_1based=[1, 2],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )


def test_run_extraction_engine_pagewise_collects_surya_sidecar(tmp_path, monkeypatch) -> None:
    image_paths = [tmp_path / f"p{idx}.png" for idx in range(1, 4)]
    source_pages = [2, 4, 6]
    for image_path in image_paths:
        _write_fixture_png(image_path)
    call_count = {"value": 0}
    progress_steps: list[tuple[int, int, int]] = []

    def fake_surya_direct(_image_paths, *, lang, work_dir, which_fn, run_cmd):
        call_count["value"] += 1
        staged_paths = list(_image_paths)
        assert [path.name for path in staged_paths] == [path.name for path in image_paths]
        assert all(path.parent == tmp_path / "work" / "source_evidence" for path in staged_paths)
        for staged_path, original_path, source_page in zip(
            staged_paths,
            image_paths,
            source_pages,
            strict=True,
        ):
            assert ocr_benchmark_mod._source_raster_identity(
                staged_path,
                source_page=source_page,
            ) == ocr_benchmark_mod._source_raster_identity(
                original_path,
                source_page=source_page,
            )
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "execution_path": "module",
                    "images": [
                        {
                            "image_name": f"{idx:04d}_{image_path.name}",
                            "pages": [
                                {
                                    "image_bbox": [0, 0, 100, 100],
                                    "text_lines": [
                                        {
                                            "text": f"surya-page-{idx}",
                                            "bbox": [0, 0, 90, 20],
                                        }
                                    ],
                                }
                            ],
                        }
                        for idx, image_path in enumerate(_image_paths, start=1)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return "aggregate", len("aggregate")

    monkeypatch.setattr(ocr_benchmark_mod, "_run_surya_direct", fake_surya_direct)

    page_texts, chars, page_errors, page_metadata = (
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_SURYA,
            image_paths,
            source_pages_1based=source_pages,
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
            progress_cb=lambda done, total, source_page: progress_steps.append(
                (done, total, source_page)
            ),
        )
    )

    assert call_count["value"] == 1
    assert page_texts == ["surya-page-1", "surya-page-2", "surya-page-3"]
    assert chars == sum(len(text) for text in page_texts)
    assert page_errors == []
    assert [item["source_page"] for item in page_metadata] == [2, 4, 6]
    assert progress_steps == [(1, 3, 2), (2, 3, 4), (3, 3, 6)]
    for metadata, image_path in zip(page_metadata, image_paths, strict=True):
        sidecar_payload = json.loads(
            Path(metadata["surya_page_lines_path"]).read_text(encoding="utf-8")
        )
        assert sidecar_payload["execution_path"] == "module"
        assert sidecar_payload["images"][0]["image_name"] == image_path.name

    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    _, pages_json_path = ocr_benchmark_mod._write_pagewise_text_artifacts(
        output_dir=tmp_path / "out",
        engine=OCR_ENGINE_SURYA,
        pdf_path=pdf_path,
        source_pages_1based=source_pages,
        page_texts=page_texts,
        aggregate_path=tmp_path / "out" / "fixture_surya.txt",
        page_metadata=page_metadata,
    )
    pages_payload = json.loads(pages_json_path.read_text(encoding="utf-8"))
    assert [item["source_page"] for item in pages_payload["pages"]] == [2, 4, 6]
    assert all(item["geometry_type"] == "surya_text_lines" for item in pages_payload["pages"])
    for row, source_page in zip(pages_payload["pages"], source_pages, strict=True):
        artifact = row["source_raster_artifact"]
        expected_path = (
            tmp_path
            / "out"
            / "surya"
            / f"page_{source_page:04d}.surya-source"
            / "source.png"
        ).resolve()
        assert artifact["path"] == str(expected_path)
        assert artifact["sha256"] == ocr_benchmark_mod._sha256_path(expected_path)
        assert artifact["bytes"] == expected_path.stat().st_size
        geometry = json.loads(
            (tmp_path / "out" / "surya" / row["geometry_file"]).read_text(encoding="utf-8")
        )
        assert geometry["images"][0]["source_raster_artifact"] == artifact


def test_run_extraction_engine_pagewise_requires_surya_sidecar_by_default(
    tmp_path, monkeypatch
) -> None:
    image_paths = [tmp_path / "p1.png"]
    _write_fixture_png(image_paths[0])

    def fake_surya_direct(_image_paths, *, lang, work_dir, which_fn, run_cmd):
        return "ok-surya", 8

    monkeypatch.setattr(ocr_benchmark_mod, "_run_surya_direct", fake_surya_direct)
    monkeypatch.delenv("UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON", raising=False)

    with pytest.raises(RuntimeError, match="surya geometry sidecar is required"):
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_SURYA,
            image_paths,
            source_pages_1based=[1],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )


def test_run_extraction_engine_pagewise_accepts_blank_surya_page(
    tmp_path,
    monkeypatch,
) -> None:
    image_paths = [tmp_path / "p1.png", tmp_path / "p2.png"]
    for image_path in image_paths:
        Image.new("RGB", (200, 300), "white").save(image_path)

    def fake_surya_direct(_image_paths, *, lang, work_dir, which_fn, run_cmd):
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "image_name": "0001_p1.png",
                            "pages": [
                                {
                                    "image_bbox": [0, 0, 200, 300],
                                    "text_lines": [
                                        {
                                            "text": "page-1",
                                            "bbox": [10, 10, 100, 30],
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "image_name": "0002_p2.png",
                            "pages": [
                                {
                                    "image_bbox": [0, 0, 200, 300],
                                    "text_lines": [],
                                }
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return "aggregate", 9

    monkeypatch.setattr(ocr_benchmark_mod, "_run_surya_direct", fake_surya_direct)
    monkeypatch.delenv("UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON", raising=False)
    progress_steps: list[tuple[int, int, int]] = []

    page_texts, chars, page_errors, page_metadata = (
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_SURYA,
            image_paths,
            source_pages_1based=[1, 2],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
            progress_cb=lambda done, total, source_page: progress_steps.append(
                (done, total, source_page)
            ),
        )
    )

    assert page_texts == ["page-1", ""]
    assert chars == len("page-1")
    assert page_errors == []
    assert page_metadata[1]["source_page"] == 2
    assert page_metadata[1]["blank_page"] is True
    assert page_metadata[1]["ocr_outcome"] == "verified_blank"
    assert page_metadata[1]["attempt_count"] == 1
    assert page_metadata[1]["alnum_chars"] == 0
    assert progress_steps == [(1, 2, 1), (2, 2, 2)]

    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    _, pages_json_path = ocr_benchmark_mod._write_pagewise_text_artifacts(
        output_dir=tmp_path / "out",
        engine=OCR_ENGINE_SURYA,
        pdf_path=pdf_path,
        source_pages_1based=[1, 2],
        page_texts=page_texts,
        aggregate_path=tmp_path / "out" / "fixture_surya.txt",
        page_metadata=page_metadata,
    )
    pages_payload = json.loads(pages_json_path.read_text(encoding="utf-8"))
    assert pages_payload["pages"][1]["blank_page"] is True


def test_run_extraction_engine_pagewise_reports_missing_surya_page_geometry(
    tmp_path,
    monkeypatch,
) -> None:
    image_paths = [tmp_path / f"p{idx}.png" for idx in range(1, 4)]
    for image_path in image_paths:
        Image.new("RGB", (200, 300), "black").save(image_path)

    def fake_surya_direct(_image_paths, *, lang, work_dir, which_fn, run_cmd):
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "execution_path": "module",
                    "images": [
                        {
                            "image_name": f"{idx:04d}_{image_path.name}",
                            "pages": (
                                [
                                    {
                                        "image_bbox": [0, 0, 200, 300],
                                        "text_lines": [
                                            {"text": f"page-{idx}", "bbox": [0, 0, 10, 10]}
                                        ],
                                    }
                                ]
                                if idx != 2
                                else [{"image_bbox": [0, 0, 200, 300], "text_lines": []}]
                            ),
                        }
                        for idx, image_path in enumerate(_image_paths, start=1)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return "aggregate", 9

    retry_calls: list[Path] = []

    def fake_surya_retry(_image_paths, *, lang, work_dir, which_fn, run_cmd):
        assert len(_image_paths) == 1
        retry_image = _image_paths[0]
        retry_calls.append(retry_image)
        with Image.open(retry_image) as image:
            assert image.size == (200, 300)
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "execution_path": "module",
                    "images": [
                        {
                            "image_name": retry_image.name,
                            "pages": [{"image_bbox": [0, 0, 200, 300], "text_lines": []}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return "", 0

    monkeypatch.setattr(ocr_benchmark_mod, "_run_surya_direct", fake_surya_direct)
    monkeypatch.setattr(ocr_benchmark_mod, "_run_surya_module_cli", fake_surya_retry)
    monkeypatch.setenv("UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON", "0")
    progress_steps: list[tuple[int, int, int]] = []

    page_texts, chars, page_errors, page_metadata = (
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_SURYA,
            image_paths,
            source_pages_1based=[1, 2, 3],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
            progress_cb=lambda done, total, source_page: progress_steps.append(
                (done, total, source_page)
            ),
        )
    )

    assert page_texts == ["page-1", "", "page-3"]
    assert chars == len("page-1") + len("page-3")
    assert [item["source_page"] for item in page_errors] == [2]
    assert [item["source_page"] for item in page_metadata] == [1, 3, 2]
    assert page_metadata[-1]["ocr_outcome"] == "zero_output"
    assert page_metadata[-1]["attempt_count"] == 3
    assert len(retry_calls) == 2
    assert progress_steps == [(1, 3, 1), (2, 3, 3)]


def test_run_extraction_engine_pagewise_collects_chandra_sidecar(tmp_path, monkeypatch) -> None:
    image_paths = [tmp_path / "p1.png", tmp_path / "p2.png"]
    for image_path in image_paths:
        _write_fixture_png(image_path)
    call_count = {"count": 0}

    progress_steps: list[tuple[int, int, int]] = []

    def fake_chandra_direct(
        _image_paths,
        *,
        lang,
        work_dir,
        which_fn,
        run_cmd,
        page_progress_cb=None,
        source_raster_identities=None,
    ):
        assert source_raster_identities is not None
        assert len(source_raster_identities) == len(_image_paths)
        call_count["count"] += 1
        assert len(_image_paths) == 2
        if page_progress_cb is not None:
            page_progress_cb(1, 2)
            page_progress_cb(2, 2)
        sidecar = work_dir / "chandra_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar_payload = {
            "images": [
                {
                    "image_name": _image_paths[0].name,
                    "pages": [{"text_lines": [{"text": "ok-chandra-1", "bbox": [0, 0, 1, 1]}]}],
                },
                {
                    "image_name": _image_paths[1].name,
                    "pages": [{"text_lines": [{"text": "ok-chandra-2", "bbox": [0, 0, 1, 1]}]}],
                },
            ]
        }
        sidecar.write_text(json.dumps(sidecar_payload, ensure_ascii=False), encoding="utf-8")
        return "ok-chandra", 22

    monkeypatch.setattr(ocr_benchmark_mod, "_run_chandra_direct", fake_chandra_direct)

    page_texts, chars, page_errors, page_metadata = (
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_CHANDRA,
            image_paths,
            source_pages_1based=[1, 2],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
            progress_cb=lambda done, total, source_page: progress_steps.append(
                (done, total, source_page)
            ),
        )
    )

    assert call_count["count"] == 1
    assert page_texts == ["ok-chandra-1", "ok-chandra-2"]
    assert chars == len("ok-chandra-1") + len("ok-chandra-2")
    assert page_errors == []
    assert len(page_metadata) == 2
    assert page_metadata[0]["source_page"] == 1
    assert page_metadata[1]["source_page"] == 2
    assert Path(page_metadata[0]["chandra_page_lines_path"]).exists()
    assert Path(page_metadata[1]["chandra_page_lines_path"]).exists()
    assert progress_steps == [(1, 2, 1), (2, 2, 2)]


def test_run_extraction_engine_pagewise_reports_partial_chandra_sidecar(
    tmp_path,
    monkeypatch,
) -> None:
    image_paths = [tmp_path / "p1.png", tmp_path / "p2.png"]
    for image_path in image_paths:
        _write_fixture_png(image_path)

    def fake_chandra_direct(_image_paths, *, work_dir, **_kwargs):
        sidecar = work_dir / "chandra_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "image_name": _image_paths[0].name,
                            "pages": [
                                {"text_lines": [{"text": "page-one", "bbox": [0, 0, 10, 10]}]}
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return "page-one aggregate", 18

    monkeypatch.setattr(ocr_benchmark_mod, "_run_chandra_direct", fake_chandra_direct)
    monkeypatch.setenv("UNISCAN_CHANDRA_REQUIRE_SIDECAR", "0")

    page_texts, chars, page_errors, page_metadata = (
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_CHANDRA,
            image_paths,
            source_pages_1based=[1, 2],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )
    )

    assert page_texts == ["page-one", ""]
    assert chars == len("page-one")
    assert [item["source_page"] for item in page_errors] == [2]
    assert [item["source_page"] for item in page_metadata] == [1]


def test_run_extraction_engine_pagewise_rejects_partial_chandra_sidecar_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    image_paths = [tmp_path / "p1.png", tmp_path / "p2.png"]
    for image_path in image_paths:
        _write_fixture_png(image_path)

    def fake_chandra_direct(_image_paths, *, work_dir, **_kwargs):
        sidecar = work_dir / "chandra_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "image_name": _image_paths[0].name,
                            "pages": [
                                {"text_lines": [{"text": "page-one", "bbox": [0, 0, 10, 10]}]}
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return "page-one aggregate", 18

    monkeypatch.setattr(ocr_benchmark_mod, "_run_chandra_direct", fake_chandra_direct)
    monkeypatch.delenv("UNISCAN_CHANDRA_REQUIRE_SIDECAR", raising=False)

    with pytest.raises(RuntimeError, match="cardinality|required for each page"):
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_CHANDRA,
            image_paths,
            source_pages_1based=[1, 2],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )


def test_run_extraction_engine_pagewise_warns_when_chandra_sidecar_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    image_paths = [tmp_path / "p1.png", tmp_path / "p2.png"]
    for image_path in image_paths:
        _write_fixture_png(image_path)

    def fake_chandra_direct(*_args, **_kwargs):
        return "aggregate chandra text", 22

    monkeypatch.setattr(ocr_benchmark_mod, "_run_chandra_direct", fake_chandra_direct)
    monkeypatch.setenv("UNISCAN_CHANDRA_REQUIRE_SIDECAR", "0")

    page_texts, chars, page_errors, page_metadata = (
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_CHANDRA,
            image_paths,
            source_pages_1based=[1, 2],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )
    )

    assert page_texts == ["aggregate chandra text", ""]
    assert chars == 22
    assert [item["source_page"] for item in page_errors] == [1, 2]
    assert all(
        item["error"] == "chandra sidecar missing or empty; aggregate text mapped to page 1"
        for item in page_errors
    )
    assert page_metadata == []


def test_run_ocr_benchmark_surfaces_chandra_sidecar_warning_in_note(
    tmp_path,
    monkeypatch,
) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [50])
    output_dir = tmp_path / "out"
    warning = "chandra sidecar missing or empty; aggregate text mapped to page 1"

    monkeypatch.setattr(
        ocr_benchmark_mod,
        "detect_ocr_engine_status",
        lambda engine_name, **_kwargs: _ready_status(engine_name, searchable_pdf=False),
    )
    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_run_extraction_engine_pagewise",
        lambda *_args, **_kwargs: (
            ["aggregate chandra text"],
            22,
            [{"source_page": 1, "image": "p1.png", "error": warning}],
            [],
        ),
    )

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_CHANDRA,),
        sample_size=1,
        dpi=100,
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].note == warning
    assert results[0].page_error_count == 1


def test_run_ocr_benchmark_deferred_candidate_is_pending_not_ok(
    tmp_path,
    monkeypatch,
) -> None:
    pdf_path = _build_sample_pdf(tmp_path, [50])
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        ocr_benchmark_mod,
        "detect_ocr_engine_status",
        lambda engine_name, **_kwargs: _ready_status(engine_name, searchable_pdf=False),
    )

    def fake_pagewise(*_args, work_dir, **_kwargs):
        sidecar = work_dir / "page_0001" / "chandra_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "image_name": "page_0001.png",
                            "ocr_outcome": "explicit_nontext",
                            "explicit_nontext": True,
                            "chandra_non_text_labels": ["figure"],
                            "attempt_count": 2,
                            "pages": [
                                {
                                    "image_bbox": [0, 0, 100, 100],
                                    "text_lines": [],
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return (
            [""],
            0,
            [
                {
                    "code": "zero_output",
                    "source_page": 1,
                    "image": "page_0001.png",
                    "error": "Chandra geometry sidecar has no text_lines",
                }
            ],
            [
                {
                    "source_page": 1,
                    "ocr_outcome": "explicit_nontext",
                    "explicit_nontext": True,
                    "attempt_count": 2,
                    "alnum_line_count": 0,
                    "alnum_chars": 0,
                    "chandra_page_lines_path": str(sidecar),
                }
            ],
        )

    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_run_extraction_engine_pagewise",
        fake_pagewise,
    )

    results = run_ocr_benchmark(
        pdf_path=pdf_path,
        output_dir=output_dir,
        engines=(OCR_ENGINE_CHANDRA,),
        sample_size=1,
        dpi=100,
        defer_empty_pages=True,
    )

    assert results[0].status == "reconciliation_pending"
    assert results[0].page_error_count == 1
    report = json.loads((output_dir / "fixture_ocr_benchmark.json").read_text(encoding="utf-8"))
    assert report["results"][0]["status"] == "reconciliation_pending"


def test_run_extraction_engine_pagewise_requires_chandra_sidecar_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "p1.png"
    _write_fixture_png(image_path)

    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_run_chandra_direct",
        lambda *_args, **_kwargs: ("aggregate chandra text", 22),
    )
    monkeypatch.delenv("UNISCAN_CHANDRA_REQUIRE_SIDECAR", raising=False)

    with pytest.raises(RuntimeError, match="chandra sidecar missing"):
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_CHANDRA,
            [image_path],
            source_pages_1based=[1],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )


def test_run_extraction_engine_pagewise_orders_chandra_columns(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "p1.png"
    _write_fixture_png(image_path)

    def fake_chandra_direct(
        _image_paths,
        *,
        lang,
        work_dir,
        which_fn,
        run_cmd,
        page_progress_cb=None,
        source_raster_identities=None,
    ):
        assert source_raster_identities is not None
        assert len(source_raster_identities) == len(_image_paths)
        sidecar = work_dir / "chandra_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "image_name": _image_paths[0].name,
                            "pages": [
                                {
                                    "text_lines": [
                                        {"text": "L1", "bbox": [20, 10, 180, 24]},
                                        {"text": "R1", "bbox": [320, 10, 480, 24]},
                                        {"text": "L2", "bbox": [20, 32, 180, 46]},
                                        {"text": "R2", "bbox": [320, 32, 480, 46]},
                                        {"text": "L3", "bbox": [20, 54, 180, 68]},
                                        {"text": "R3", "bbox": [320, 54, 480, 68]},
                                    ]
                                }
                            ],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return "ignored aggregate", 16

    monkeypatch.setattr(ocr_benchmark_mod, "_run_chandra_direct", fake_chandra_direct)

    page_texts, chars, page_errors, page_metadata = (
        ocr_benchmark_mod._run_extraction_engine_pagewise(
            OCR_ENGINE_CHANDRA,
            [image_path],
            source_pages_1based=[1],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )
    )

    assert page_texts == ["L1\nL2\nL3\nR1\nR2\nR3"]
    assert chars == len(page_texts[0])
    assert page_errors == []
    assert len(page_metadata) == 1


def test_write_pagewise_text_artifacts_copies_chandra_geometry(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    aggregate_path = tmp_path / "doc_chandra.txt"
    sidecar_src = tmp_path / "chandra_page_lines.json"
    sidecar_src.write_text('{"images":[]}', encoding="utf-8")
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    total_chars, pages_json_path = ocr_benchmark_mod._write_pagewise_text_artifacts(
        output_dir=output_dir,
        engine=OCR_ENGINE_CHANDRA,
        pdf_path=pdf_path,
        source_pages_1based=[1],
        page_texts=["line-1"],
        aggregate_path=aggregate_path,
        page_metadata=[{"source_page": 1, "chandra_page_lines_path": str(sidecar_src)}],
    )

    assert total_chars == len("line-1")
    assert pages_json_path.exists()
    copied = output_dir / OCR_ENGINE_CHANDRA / "page_0001.chandra.json"
    assert copied.exists()
    payload = json.loads(pages_json_path.read_text(encoding="utf-8"))
    assert payload["pages"][0]["geometry_file"] == "page_0001.chandra.json"
    assert payload["pages"][0]["geometry_type"] == "chandra_text_lines"


def test_configure_chandra_runtime_device_prefers_cuda(monkeypatch) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    fake_torch = SimpleNamespace(__version__="9.9.9", cuda=_Cuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.setenv("UNISCAN_CHANDRA_DEVICE_POLICY", "legacy")
    monkeypatch.setenv("UNISCAN_CHANDRA_PREFER_GPU", "1")
    monkeypatch.delenv("UNISCAN_CHANDRA_REQUIRE_GPU", raising=False)

    device = ocr_benchmark_mod._configure_chandra_runtime_device()

    assert device == "cuda:0"
    assert os.environ.get("TORCH_DEVICE") == "cuda:0"


def test_configure_chandra_runtime_device_raises_when_gpu_required_without_cuda(
    monkeypatch,
) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = SimpleNamespace(__version__="9.9.9", cuda=_Cuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.setenv("UNISCAN_CHANDRA_PREFER_GPU", "1")
    monkeypatch.setenv("UNISCAN_CHANDRA_REQUIRE_GPU", "1")

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        ocr_benchmark_mod._configure_chandra_runtime_device()


def test_configure_chandra_runtime_device_rejects_cpu_when_gpu_required(monkeypatch) -> None:
    monkeypatch.setenv("TORCH_DEVICE", "cpu")
    monkeypatch.delenv("UNISCAN_CHANDRA_REQUIRE_GPU", raising=False)

    with pytest.raises(RuntimeError, match="TORCH_DEVICE='cpu'"):
        ocr_benchmark_mod._configure_chandra_runtime_device()


def test_configure_surya_runtime_device_requires_cuda_by_default(monkeypatch) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = SimpleNamespace(__version__="9.9.9", cuda=_Cuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.delenv("UNISCAN_SURYA_REQUIRE_GPU", raising=False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        ocr_benchmark_mod._configure_surya_runtime_device()


def test_configure_surya_runtime_device_sets_cuda_when_required(monkeypatch) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

    fake_torch = SimpleNamespace(__version__="9.9.9", cuda=_Cuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.setenv("UNISCAN_SURYA_REQUIRE_GPU", "1")

    device = ocr_benchmark_mod._configure_surya_runtime_device()

    assert device == "cuda:0"
    assert os.environ.get("TORCH_DEVICE") == "cuda:0"


def test_chandra_chunk_lines_preserves_explicit_breaks() -> None:
    raw = "<p>FIRST LINE<br/>SECOND LINE</p><div>THIRD LINE</div>"
    lines = ocr_benchmark_mod._chandra_chunk_lines(raw)
    assert lines == ["FIRST LINE", "SECOND LINE", "THIRD LINE"]
    assert ocr_benchmark_mod._chandra_chunk_lines(0) == ["0"]


def test_chandra_graphic_chunk_lines_keeps_only_visible_text_tags() -> None:
    raw = (
        '<img alt="invented alt text"/>'
        "<div>Detailed visual description</div>"
        "<div><p>Nested invented description</p></div>"
        "<p>VISIBLE<br/>TRANSCRIPT</p>"
        "<svg><text>diagram description</text></svg>"
    )

    assert ocr_benchmark_mod._chandra_graphic_chunk_lines(raw) == [
        "VISIBLE",
        "TRANSCRIPT",
    ]
    assert (
        ocr_benchmark_mod._chandra_graphic_chunk_lines(
            '<img alt="invented"/><div>description only</div>'
        )
        == []
    )
    assert ocr_benchmark_mod._chandra_graphic_chunk_ignored_lines(raw) == [
        "Detailed visual description",
        "Nested invented description",
        "diagram description",
    ]


def test_chandra_expand_chunk_to_line_boxes_splits_rows() -> None:
    lines = [
        "ONE VERY LONG LINE THAT SHOULD BE WRAPPED INTO MULTIPLE SEGMENTS "
        "TO IMPROVE SEARCHABLE PDF SELECTION QUALITY",
        "TAIL LINE",
    ]
    placements = ocr_benchmark_mod._chandra_expand_chunk_to_line_boxes(
        lines=lines,
        bbox=[100.0, 200.0, 500.0, 320.0],
    )

    assert len(placements) >= 3
    ys = [float(item["bbox"][1]) for item in placements]
    assert ys == sorted(ys)
    assert all(float(item["bbox"][0]) == pytest.approx(100.0) for item in placements)
    assert all(float(item["bbox"][2]) == pytest.approx(500.0) for item in placements)
    assert any("TAIL LINE" in str(item["text"]) for item in placements)


def test_run_chandra_module_accepts_verified_blank_page(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (200, 300), "white").save(image_path)
    progress_steps: list[tuple[int, int]] = []

    class FakeInferenceManager:
        def __init__(self, *, method: str) -> None:
            assert method == "hf"

        def generate(
            self,
            _batch,
            *,
            include_images: bool,
            include_headers_footers: bool,
        ):
            assert include_images is False
            assert include_headers_footers is False
            return [
                SimpleNamespace(
                    chunks=[{"label": "blank-page", "content": ""}],
                    markdown="",
                )
            ]

    model_module = ModuleType("chandra.model")
    model_module.InferenceManager = FakeInferenceManager
    schema_module = ModuleType("chandra.model.schema")
    schema_module.BatchInputItem = lambda **kwargs: kwargs
    prompts_module = ModuleType("chandra.prompts")
    prompts_module.PROMPT_MAPPING = {
        "ocr_layout": "fake layout prompt",
        "ocr": "fake plain prompt",
    }
    input_module = ModuleType("chandra.input")
    input_module.load_image = _load_fixture_chandra_image
    chandra_module = ModuleType("chandra")
    chandra_module.model = model_module

    monkeypatch.setitem(sys.modules, "chandra", chandra_module)
    monkeypatch.setitem(sys.modules, "chandra.model", model_module)
    monkeypatch.setitem(sys.modules, "chandra.model.schema", schema_module)
    monkeypatch.setitem(sys.modules, "chandra.prompts", prompts_module)
    monkeypatch.setitem(sys.modules, "chandra.input", input_module)
    monkeypatch.setattr(ocr_benchmark_mod, "_ensure_chandra_cache_ready", lambda: None)
    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_configure_chandra_runtime_device",
        lambda: "cuda:0",
    )
    monkeypatch.setenv("UNISCAN_CHANDRA_REQUIRE_GPU", "0")

    text, chars = ocr_benchmark_mod._run_chandra_module(
        [image_path],
        lang="rus",
        work_dir=tmp_path / "work",
        page_progress_cb=lambda done, total: progress_steps.append((done, total)),
    )

    assert text == ""
    assert chars == 0
    assert progress_steps == [(1, 1)]
    sidecar = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )
    assert sidecar["images"][0]["pages"][0]["text_lines"] == []


def test_run_chandra_direct_disables_cli_fallback_by_default(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "p1.png"
    _write_fixture_png(image_path)

    def fail_module(*_args, **_kwargs):
        raise RuntimeError("module failed")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("CLI fallback must stay disabled by default")

    monkeypatch.setattr(ocr_benchmark_mod, "_run_chandra_module", fail_module)
    monkeypatch.setattr(ocr_benchmark_mod, "_run_chandra_cli", fail_if_called)
    monkeypatch.delenv("UNISCAN_CHANDRA_ALLOW_CLI_FALLBACK", raising=False)

    with pytest.raises(RuntimeError, match="CLI fallback is disabled"):
        ocr_benchmark_mod._run_chandra_direct(
            [image_path],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )


def test_run_chandra_direct_allows_cli_fallback_when_enabled(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "p1.png"
    _write_fixture_png(image_path)

    def fail_module(*_args, **_kwargs):
        raise RuntimeError("module failed")

    monkeypatch.setattr(ocr_benchmark_mod, "_run_chandra_module", fail_module)
    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_run_chandra_cli",
        lambda *_args, **_kwargs: ("cli-ok", 6),
    )
    monkeypatch.setenv("UNISCAN_CHANDRA_ALLOW_CLI_FALLBACK", "1")

    text, chars = ocr_benchmark_mod._run_chandra_direct(
        [image_path],
        lang="rus",
        work_dir=tmp_path / "work",
        which_fn=lambda _name: None,
        run_cmd=lambda *_args, **_kwargs: None,
    )
    assert text == "cli-ok"
    assert chars == 6


def test_run_surya_direct_disables_text_fallback_by_default(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "p1.png"
    _write_fixture_png(image_path)

    def fail_module(*_args, **_kwargs):
        raise RuntimeError("module failed")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Text-only fallback must stay disabled by default")

    monkeypatch.setattr(ocr_benchmark_mod, "_run_surya_module_cli", fail_module)
    monkeypatch.setattr(ocr_benchmark_mod, "_run_text_engine_from_cli", fail_if_called)
    monkeypatch.setattr(ocr_benchmark_mod, "_ensure_surya_cache_ready", lambda: None)
    monkeypatch.delenv("UNISCAN_SURYA_ALLOW_TEXT_FALLBACK", raising=False)
    monkeypatch.setenv("UNISCAN_SURYA_REQUIRE_GPU", "0")

    with pytest.raises(RuntimeError, match="Text-only fallback is disabled"):
        ocr_benchmark_mod._run_surya_direct(
            [image_path],
            lang="rus",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )


def test_run_surya_direct_allows_text_fallback_when_enabled(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "p1.png"
    _write_fixture_png(image_path)

    def fail_module(*_args, **_kwargs):
        raise RuntimeError("module failed")

    monkeypatch.setattr(ocr_benchmark_mod, "_run_surya_module_cli", fail_module)
    monkeypatch.setattr(
        ocr_benchmark_mod,
        "_run_text_engine_from_cli",
        lambda *_args, **_kwargs: ("cli-ok", 6),
    )
    monkeypatch.setattr(ocr_benchmark_mod, "_ensure_surya_cache_ready", lambda: None)
    monkeypatch.setenv("UNISCAN_SURYA_ALLOW_TEXT_FALLBACK", "1")
    monkeypatch.setenv("UNISCAN_SURYA_REQUIRE_GPU", "0")

    text, chars = ocr_benchmark_mod._run_surya_direct(
        [image_path],
        lang="rus",
        work_dir=tmp_path / "work",
        which_fn=lambda _name: None,
        run_cmd=lambda *_args, **_kwargs: None,
    )
    assert text == "cli-ok"
    assert chars == 6


def test_surya_module_cli_uses_only_staged_inputs(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page_0001.png"
    _write_fixture_png(image_path)
    work_dir = tmp_path / "work"
    stale_input_dir = work_dir / "surya_input"
    stale_input_dir.mkdir(parents=True)
    (stale_input_dir / "stale.png").write_bytes(b"stale")

    surya_module = ModuleType("surya")
    scripts_module = ModuleType("surya.scripts")
    ocr_text_module = ModuleType("surya.scripts.ocr_text")

    class _DummyCli:
        @staticmethod
        def main(*, args, standalone_mode=False):
            assert standalone_mode is False
            input_dir = Path(args[0])
            output_root = Path(args[2])
            payload = {
                name: [{"text_lines": [{"text": f"TEXT::{name}", "bbox": [0, 0, 100, 20]}]}]
                for name in sorted(path.name for path in input_dir.iterdir() if path.is_file())
            }
            # Simulate unrelated page accidentally present in raw model output.
            payload["foreign_page.png"] = [{"text_lines": [{"text": "FOREIGN"}]}]
            result_file = output_root / input_dir.name / "results.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    ocr_text_module.ocr_text_cli = _DummyCli
    monkeypatch.setitem(sys.modules, "surya", surya_module)
    monkeypatch.setitem(sys.modules, "surya.scripts", scripts_module)
    monkeypatch.setitem(sys.modules, "surya.scripts.ocr_text", ocr_text_module)

    text, chars = ocr_benchmark_mod._run_surya_module_cli(
        [image_path],
        lang="rus",
        work_dir=work_dir,
        run_cmd=lambda *_args, **_kwargs: None,
    )

    assert "TEXT::page_0001.png" in text
    assert "FOREIGN" not in text
    assert chars == len(text)
    sidecar_payload = json.loads((work_dir / "surya_page_lines.json").read_text(encoding="utf-8"))
    assert sidecar_payload["execution_path"] == "module"
    assert "stale.png" not in text


def test_surya_cli_fallback_cannot_reuse_partial_module_output(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    _write_fixture_png(image_path)
    work_dir = tmp_path / "work"

    surya_module = ModuleType("surya")
    scripts_module = ModuleType("surya.scripts")
    ocr_text_module = ModuleType("surya.scripts.ocr_text")

    class _FailingModuleCli:
        @staticmethod
        def main(*, args, standalone_mode=False):
            assert standalone_mode is False
            input_dir = Path(args[0])
            output_root = Path(args[2])
            result_file = output_root / input_dir.name / "results.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                json.dumps(
                    {image_path.name: [{"text_lines": [{"text": "STALE", "bbox": [0, 0, 10, 10]}]}]}
                ),
                encoding="utf-8",
            )
            raise RuntimeError("module failed after partial output")

    ocr_text_module.ocr_text_cli = _FailingModuleCli
    monkeypatch.setitem(sys.modules, "surya", surya_module)
    monkeypatch.setitem(sys.modules, "surya.scripts", scripts_module)
    monkeypatch.setitem(sys.modules, "surya.scripts.ocr_text", ocr_text_module)

    with pytest.raises(RuntimeError, match="did not produce results file"):
        ocr_benchmark_mod._run_surya_module_cli(
            [image_path],
            lang="rus",
            work_dir=work_dir,
            which_fn=lambda _name: "surya_ocr",
            run_cmd=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="",
            ),
        )

    assert not (work_dir / "surya_page_lines.json").exists()


@pytest.mark.skipif(not FIXTURE_PDF.exists(), reason="external OCR fixture is not available")
def test_run_ocr_benchmark_uses_external_fixture_smoke(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "out"

    def fake_sample(_page_count: int, *, sample_size: int = 5) -> list[int]:
        assert sample_size == 1
        return [0]

    def fake_status(engine_name: str, **_kwargs):
        return _ready_status(engine_name, searchable_pdf=False)

    def fake_paddleocr(image_paths, *, lang):
        assert len(image_paths) == 1
        return f"{lang}:fixture", 7

    monkeypatch.setattr("uniscan.ocr.benchmark.sample_pdf_page_indices", fake_sample)
    monkeypatch.setattr("uniscan.ocr.benchmark.detect_ocr_engine_status", fake_status)
    monkeypatch.setattr("uniscan.ocr.benchmark._run_paddleocr_direct", fake_paddleocr)

    results = run_ocr_benchmark(
        pdf_path=FIXTURE_PDF,
        output_dir=output_dir,
        engines=(OCR_ENGINE_PADDLEOCR,),
        sample_size=1,
        dpi=72,
        lang="eng",
    )

    assert len(results) == 1
    assert results[0].status == "ok"
    assert results[0].sample_pages == [1]
    assert results[0].artifact_path is not None
    assert Path(results[0].artifact_path).exists()
    assert (output_dir / "Imaging Edge Mobile_paddleocr_uvdoc_ocr_benchmark.json").exists()


def test_cli_benchmark_ocr_uses_runner_and_returns_success(monkeypatch, tmp_path, capsys) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run_ocr_benchmark(**kwargs):
        assert kwargs["pdf_path"] == pdf_path
        assert kwargs["output_dir"] == output_dir
        assert kwargs["sample_size"] == 5
        assert kwargs["defer_empty_pages"] is False
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PADDLEOCR,
                status="ok",
                sample_pages=[1],
                elapsed_seconds=1.23,
                artifact_path=str(output_dir / "fixture_paddleocr.txt"),
                text_chars=7,
                memory_delta_mb=1.0,
                error=None,
                note=None,
            )
        ]

    def fake_summary(results):
        assert len(results) == 1
        return "paddleocr ok"

    monkeypatch.setattr("uniscan.cli.run_ocr_benchmark", fake_run_ocr_benchmark)
    monkeypatch.setattr("uniscan.cli.summarize_ocr_benchmark", fake_summary)

    exit_code = main(["benchmark-ocr", "--pdf", str(pdf_path), "--output", str(output_dir)])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "paddleocr ok" in stdout


def test_cli_benchmark_ocr_rejects_manual_internal_reconciliation_token(
    monkeypatch,
    tmp_path,
) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.delenv("UNISCAN_INTERNAL_RECONCILIATION_TOKEN", raising=False)
    monkeypatch.setattr(
        "uniscan.cli.run_ocr_benchmark",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual internal context must fail before benchmark")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "benchmark-ocr",
                "--pdf",
                str(pdf_path),
                "--output",
                str(output_dir),
                "--internal-reconciliation-token",
                "0" * 32,
            ]
        )

    assert exc_info.value.code == 2


def test_cli_benchmark_ocr_parses_pages(monkeypatch, tmp_path, capsys) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run_ocr_benchmark(**kwargs):
        assert kwargs["page_numbers"] == (3, 9)
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PADDLEOCR,
                status="ok",
                sample_pages=[3, 9],
                elapsed_seconds=1.23,
                artifact_path=str(output_dir / "fixture_paddleocr.txt"),
                text_chars=7,
                memory_delta_mb=1.0,
                error=None,
                note=None,
            )
        ]

    monkeypatch.setattr("uniscan.cli.run_ocr_benchmark", fake_run_ocr_benchmark)
    monkeypatch.setattr("uniscan.cli.summarize_ocr_benchmark", lambda _results: "ok")

    exit_code = main(
        [
            "benchmark-ocr",
            "--pdf",
            str(pdf_path),
            "--output",
            str(output_dir),
            "--pages",
            "3,9",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "ok" in stdout


def test_cli_benchmark_ocr_strict_fails_when_any_engine_not_ok(
    monkeypatch, tmp_path, capsys
) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_run_ocr_benchmark(**_kwargs):
        return [
            SimpleNamespace(
                engine=OCR_ENGINE_PADDLEOCR,
                status="ok",
                sample_pages=[1],
                elapsed_seconds=1.0,
                artifact_path=str(output_dir / "ok.txt"),
                text_chars=10,
                memory_delta_mb=1.1,
                error=None,
                note=None,
            ),
            SimpleNamespace(
                engine=OCR_ENGINE_SURYA,
                status="error",
                sample_pages=[1],
                elapsed_seconds=1.0,
                artifact_path=str(output_dir / "err.txt"),
                text_chars=0,
                memory_delta_mb=1.2,
                error="broken",
                note=None,
            ),
        ]

    monkeypatch.setattr("uniscan.cli.run_ocr_benchmark", fake_run_ocr_benchmark)
    monkeypatch.setattr(
        "uniscan.cli.summarize_ocr_benchmark", lambda results: f"rows={len(results)}"
    )

    exit_code = main(
        ["benchmark-ocr", "--pdf", str(pdf_path), "--output", str(output_dir), "--strict"]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 1
    assert "rows=2" in stdout
