from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest
from PIL import Image

import uniscan.ocr.benchmark as benchmark
from uniscan.ocr import OCR_ENGINE_SURYA


def _nonblank_image(path: Path) -> None:
    image = Image.new("RGB", (120, 80), (80, 80, 80))
    for x in range(20, 100):
        for y in range(20, 60):
            image.putpixel((x, y), (120, 120, 120))
    image.save(path)


def _install_fake_chandra(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generate: Callable[..., list[Any]],
) -> None:
    class FakeInferenceManager:
        def __init__(self, *, method: str) -> None:
            assert method == "hf"

        def generate(self, batch, *, include_images: bool, include_headers_footers: bool):
            assert include_images is False
            assert include_headers_footers is False
            return generate(batch)

    model_module = ModuleType("chandra.model")
    model_module.InferenceManager = FakeInferenceManager
    schema_module = ModuleType("chandra.model.schema")
    schema_module.BatchInputItem = lambda **kwargs: kwargs
    input_module = ModuleType("chandra.input")
    input_module.load_image = lambda path: Image.open(path).convert("RGB")
    chandra_module = ModuleType("chandra")
    chandra_module.model = model_module
    monkeypatch.setitem(sys.modules, "chandra", chandra_module)
    monkeypatch.setitem(sys.modules, "chandra.model", model_module)
    monkeypatch.setitem(sys.modules, "chandra.model.schema", schema_module)
    monkeypatch.setitem(sys.modules, "chandra.input", input_module)
    monkeypatch.setattr(benchmark, "_ensure_chandra_cache_ready", lambda: None)
    monkeypatch.setattr(benchmark, "_configure_chandra_runtime_device", lambda: "cuda:0")
    monkeypatch.setenv("UNISCAN_CHANDRA_REQUIRE_GPU", "0")


def test_chandra_zero_output_retries_once_with_same_size_and_recovers_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    generated_sizes: list[tuple[int, int]] = []
    autocontrast_calls: list[tuple[tuple[int, int], int]] = []
    real_autocontrast = benchmark.ImageOps.autocontrast

    def spy_autocontrast(image, cutoff=0, **kwargs):
        autocontrast_calls.append((image.size, cutoff))
        return real_autocontrast(image, cutoff=cutoff, **kwargs)

    def generate(batch):
        generated_sizes.append(batch[0]["image"].size)
        if len(generated_sizes) == 1:
            return [SimpleNamespace(chunks=[], markdown="")]
        return [
            SimpleNamespace(
                chunks=[{"label": "Text", "content": "RECOVERED", "bbox": [1, 2, 50, 20]}],
                markdown="",
            )
        ]

    _install_fake_chandra(monkeypatch, generate=generate)
    monkeypatch.setattr(benchmark.ImageOps, "autocontrast", spy_autocontrast)

    text, chars = benchmark._run_chandra_module(
        [image_path],
        lang="eng",
        work_dir=tmp_path / "work",
    )

    assert text == "RECOVERED"
    assert chars == len("RECOVERED")
    assert generated_sizes == [(120, 80), (120, 80)]
    assert autocontrast_calls == [((120, 80), 1)]
    sidecar = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )
    image_evidence = sidecar["images"][0]
    assert image_evidence["ocr_outcome"] == "text"
    assert image_evidence["attempt_count"] == 2
    assert image_evidence["retry_preprocessing"] == "autocontrast-cutoff-1"


def test_chandra_retry_records_explicit_graphics_after_two_zero_text_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    calls = 0

    def generate(_batch):
        nonlocal calls
        calls += 1
        return [
            SimpleNamespace(
                chunks=[{"label": "Figure", "content": "", "bbox": [0, 0, 120, 80]}],
                markdown="",
            )
        ]

    _install_fake_chandra(monkeypatch, generate=generate)

    text, chars = benchmark._run_chandra_module(
        [image_path],
        lang="eng",
        work_dir=tmp_path / "work",
    )

    assert (text, chars) == ("", 0)
    assert calls == 2
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["ocr_outcome"] == "explicit_nontext"
    assert evidence["explicit_nontext"] is True
    assert evidence["chandra_non_text_labels"] == ["figure"]
    assert evidence["pages"][0]["text_lines"] == []



def test_chandra_preserves_original_explicit_label_when_retry_has_no_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    calls = 0

    def generate(_batch):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                SimpleNamespace(
                    chunks=[{"label": "Figure", "content": ""}],
                    markdown="",
                )
            ]
        return [SimpleNamespace(chunks=[], markdown="")]

    _install_fake_chandra(monkeypatch, generate=generate)

    text, chars = benchmark._run_chandra_module(
        [image_path],
        lang="eng",
        work_dir=tmp_path / "work",
    )

    assert (text, chars) == ("", 0)
    assert calls == 2
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["ocr_outcome"] == "explicit_nontext"
    assert evidence["chandra_non_text_labels"] == ["figure"]

def test_chandra_exception_does_not_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    calls = 0

    def generate(_batch):
        nonlocal calls
        calls += 1
        raise MemoryError("oom")

    _install_fake_chandra(monkeypatch, generate=generate)

    with pytest.raises(MemoryError, match="oom"):
        benchmark._run_chandra_module(
            [image_path],
            lang="eng",
            work_dir=tmp_path / "work",
        )
    assert calls == 1


def test_chandra_zero_result_never_invokes_cli_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    monkeypatch.setattr(benchmark, "_run_chandra_module", lambda *_args, **_kwargs: ("", 0))
    monkeypatch.setattr(
        benchmark,
        "_run_chandra_cli",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("CLI fallback must not run for zero output")
        ),
    )

    assert benchmark._run_chandra_direct(
        [image_path],
        lang="eng",
        work_dir=tmp_path / "work",
        which_fn=lambda _name: None,
        run_cmd=lambda *_args, **_kwargs: None,
    ) == ("", 0)


def test_chandra_standalone_explicit_graphics_stays_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)

    def fake_direct(image_paths, *, lang, work_dir, which_fn, run_cmd, page_progress_cb):
        sidecar = work_dir / "chandra_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "image_name": image_paths[0].name,
                            "ocr_outcome": "explicit_nontext",
                            "explicit_nontext": True,
                            "attempt_count": 2,
                            "pages": [
                                {
                                    "image_bbox": [0, 0, 120, 80],
                                    "text_lines": [],
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return "", 0

    monkeypatch.setattr(benchmark, "_run_chandra_direct", fake_direct)
    monkeypatch.delenv("UNISCAN_CHANDRA_REQUIRE_GEOMETRY_JSON", raising=False)

    with pytest.raises(RuntimeError, match="chandra geometry sidecar is required"):
        benchmark._run_extraction_engine_pagewise(
            "chandra",
            [image_path],
            source_pages_1based=[1],
            lang="eng",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
        )


def _install_initial_surya_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_direct(image_paths, *, lang, work_dir, which_fn, run_cmd):
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "image_name": image_paths[0].name,
                            "pages": [
                                {"image_bbox": [0, 0, 120, 80], "text_lines": []}
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return "", 0

    monkeypatch.setattr(benchmark, "_run_surya_direct", fake_direct)
    monkeypatch.setattr(
        benchmark,
        "_run_text_engine_from_cli",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("text fallback must not run during retry")
        ),
    )


def _write_surya_retry_sidecar(
    *,
    image_paths: list[Path],
    work_dir: Path,
    text: str,
) -> None:
    retry_image = image_paths[0]
    with Image.open(retry_image) as image:
        assert image.size == (120, 80)
    sidecar = work_dir / "surya_page_lines.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_name": retry_image.name,
                        "pages": [
                            {
                                "image_bbox": [0, 0, 120, 80],
                                "text_lines": (
                                    [{"text": text, "bbox": [1, 2, 50, 20]}]
                                    if text
                                    else []
                                ),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _run_surya_pagewise(tmp_path: Path, *, defer_empty_pages: bool):
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    return benchmark._run_extraction_engine_pagewise(
        OCR_ENGINE_SURYA,
        [image_path],
        source_pages_1based=[1],
        lang="eng",
        work_dir=tmp_path / "work",
        which_fn=lambda _name: None,
        run_cmd=lambda *_args, **_kwargs: None,
        defer_empty_pages=defer_empty_pages,
    )


def test_surya_zero_output_retry_succeeds_once(tmp_path: Path, monkeypatch) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_calls = 0

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        nonlocal retry_calls
        retry_calls += 1
        _write_surya_retry_sidecar(
            image_paths=image_paths,
            work_dir=work_dir,
            text="RECOVERED",
        )
        return "RECOVERED", len("RECOVERED")

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    page_texts, chars, page_errors, page_metadata = _run_surya_pagewise(
        tmp_path,
        defer_empty_pages=True,
    )

    assert retry_calls == 1
    assert page_texts == ["RECOVERED"]
    assert chars == len("RECOVERED")
    assert page_errors == []
    assert page_metadata[0]["attempt_count"] == 2
    assert page_metadata[0]["retry_preprocessing"] == "autocontrast-cutoff-1"


def test_surya_zero_output_retry_failure_is_durable_for_reconciliation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_calls = 0

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        nonlocal retry_calls
        retry_calls += 1
        _write_surya_retry_sidecar(image_paths=image_paths, work_dir=work_dir, text="")
        return "", 0

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    page_texts, chars, page_errors, page_metadata = _run_surya_pagewise(
        tmp_path,
        defer_empty_pages=True,
    )

    assert retry_calls == 1
    assert (page_texts, chars) == ([""], 0)
    assert [item["source_page"] for item in page_errors] == [1]
    assert page_metadata[0]["ocr_outcome"] == "zero_output"
    assert page_metadata[0]["attempt_count"] == 2


def test_surya_retry_exception_propagates_without_another_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_calls = 0

    def retry(*_args, **_kwargs):
        nonlocal retry_calls
        retry_calls += 1
        raise MemoryError("oom")

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    with pytest.raises(MemoryError, match="oom"):
        _run_surya_pagewise(tmp_path, defer_empty_pages=True)
    assert retry_calls == 1


def test_surya_standalone_zero_output_stays_fail_closed(tmp_path: Path, monkeypatch) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_calls = 0

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        nonlocal retry_calls
        retry_calls += 1
        _write_surya_retry_sidecar(image_paths=image_paths, work_dir=work_dir, text="")
        return "", 0

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)
    monkeypatch.delenv("UNISCAN_SURYA_REQUIRE_GEOMETRY_JSON", raising=False)

    with pytest.raises(RuntimeError, match="geometry sidecar is required"):
        _run_surya_pagewise(tmp_path, defer_empty_pages=False)
    assert retry_calls == 1
