from __future__ import annotations

import hashlib
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
                    "execution_path": "module",
                    "images": [
                        {
                            "image_name": image_paths[0].name,
                            "pages": [{"image_bbox": [0, 0, 120, 80], "text_lines": []}],
                        }
                    ],
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
                "execution_path": "module",
                "images": [
                    {
                        "image_name": retry_image.name,
                        "pages": [
                            {
                                "image_bbox": [0, 0, 120, 80],
                                "text_lines": (
                                    [{"text": text, "bbox": [1, 2, 50, 20]}] if text else []
                                ),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_malformed_raw_surya_sidecar(
    *,
    image_path: Path,
    work_dir: Path,
    defect: str,
) -> None:
    zero_image: dict[str, Any] = {
        "image_name": image_path.name,
        "pages": [{"image_bbox": [0, 0, 120, 80], "text_lines": []}],
    }
    text_image: dict[str, Any] = {
        "image_name": image_path.name,
        "pages": [
            {
                "image_bbox": [0, 0, 120, 80],
                "text_lines": [
                    {"text": "SOLD OUT", "bbox": [1, 2, 50, 20]},
                ],
            }
        ],
    }
    images: list[object]
    if defect == "duplicate-text-zero":
        images = [text_image, zero_image]
    elif defect == "duplicate-zero-text":
        images = [zero_image, text_image]
    elif defect == "duplicate-zero":
        images = [zero_image, dict(zero_image)]
    elif defect == "nonobject":
        images = ["not-an-image-object"]
    elif defect == "unexpected-extra":
        images = [
            zero_image,
            {
                "image_name": "unexpected.png",
                "pages": [{"image_bbox": [0, 0, 120, 80], "text_lines": []}],
            },
        ]
    elif defect == "extra-page":
        images = [zero_image]
        zero_image["pages"] = [
            {"image_bbox": [0, 0, 120, 80], "text_lines": []},
            {"image_bbox": [0, 0, 120, 80], "text_lines": []},
        ]
    elif defect == "malformed-page":
        images = [zero_image]
        zero_image["pages"] = ["not-a-page-object"]
    else:
        raise AssertionError(f"unknown raw sidecar defect: {defect}")

    sidecar = work_dir / "surya_page_lines.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps({"execution_path": "module", "images": images}),
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


def test_surya_third_retry_uses_binary_otsu_and_recovers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_images: list[Path] = []
    autocontrast_calls: list[tuple[str, tuple[int, int], int]] = []
    real_autocontrast = benchmark.ImageOps.autocontrast

    def spy_autocontrast(image, cutoff=0, **kwargs):
        autocontrast_calls.append((image.mode, image.size, cutoff))
        return real_autocontrast(image, cutoff=cutoff, **kwargs)

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        retry_images.append(image_paths[0])
        if len(retry_images) == 1:
            _write_surya_retry_sidecar(
                image_paths=image_paths,
                work_dir=work_dir,
                text="",
            )
            return "", 0

        with Image.open(image_paths[0]) as image:
            assert image.size == (120, 80)
            assert image.mode == "L"
            assert set(image.getdata()) == {0, 255}
            assert image.getpixel((0, 0)) == 0
            assert image.getpixel((40, 40)) == 255
        _write_surya_retry_sidecar(
            image_paths=image_paths,
            work_dir=work_dir,
            text="SOLD OUT",
        )
        return "SOLD OUT", len("SOLD OUT")

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)
    monkeypatch.setattr(benchmark.ImageOps, "autocontrast", spy_autocontrast)

    page_texts, chars, page_errors, page_metadata = _run_surya_pagewise(
        tmp_path,
        defer_empty_pages=True,
    )

    assert len(retry_images) == 2
    assert (page_texts, chars) == (["SOLD OUT"], len("SOLD OUT"))
    assert page_errors == []
    assert page_metadata[0]["attempt_count"] == 3
    assert page_metadata[0]["retry_preprocessing"] == "grayscale-autocontrast-otsu-v1"
    assert autocontrast_calls == [
        ("RGB", (120, 80), 1),
        ("L", (120, 80), 1),
    ]
    assert page_metadata[0]["selected_attempt"] == 3
    assert page_metadata[0]["retry_policy"] == ("original+autocontrast-cutoff-1+otsu-max3-v2")
    attempts = page_metadata[0]["attempt_history"]
    assert [item["attempt"] for item in attempts] == [1, 2, 3]
    assert [item["ocr_outcome"] for item in attempts] == [
        "zero_output",
        "zero_output",
        "text",
    ]
    assert [item["preprocessing"] for item in attempts] == [
        "original",
        "autocontrast-cutoff-1",
        "grayscale-autocontrast-otsu-v1",
    ]
    assert attempts[2]["image_size"] == [120, 80]
    assert attempts[2]["otsu_threshold"] == 0
    assert len(attempts[2]["image_sha256"]) == 64
    assert all(len(item["sidecar_sha256"]) == 64 for item in attempts)
    durable = json.loads(
        Path(page_metadata[0]["surya_page_lines_path"]).read_text(encoding="utf-8")
    )["images"][0]
    assert durable["attempt_history"] == attempts
    assert durable["selected_attempt"] == 3
    assert durable["retry_policy"] == page_metadata[0]["retry_policy"]

    output_dir = tmp_path / "output"
    benchmark._write_pagewise_text_artifacts(
        output_dir=output_dir,
        engine=OCR_ENGINE_SURYA,
        pdf_path=tmp_path / "document.pdf",
        source_pages_1based=[1],
        page_texts=page_texts,
        aggregate_path=output_dir / "document_surya.txt",
        page_metadata=page_metadata,
        page_errors=page_errors,
    )
    pages_payload = json.loads((output_dir / "surya" / "pages.json").read_text(encoding="utf-8"))
    durable_history = pages_payload["pages"][0]["attempt_history"]
    assert [item["attempt"] for item in durable_history] == [1, 2, 3]
    for item in durable_history:
        image_path = Path(item["image_path"])
        sidecar_path = Path(item["sidecar_path"])
        assert image_path.is_file()
        assert sidecar_path.is_file()
        assert image_path.is_relative_to(output_dir / "surya")
        assert sidecar_path.is_relative_to(output_dir / "surya")
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == item["image_sha256"]
        assert hashlib.sha256(sidecar_path.read_bytes()).hexdigest() == item["sidecar_sha256"]
        assert image_path.stat().st_size == item["image_bytes"]
        assert sidecar_path.stat().st_size == item["sidecar_bytes"]
    persisted_geometry = json.loads(
        (output_dir / "surya" / "page_0001.surya.json").read_text(encoding="utf-8")
    )
    persisted_image = persisted_geometry["images"][0]
    assert persisted_image["attempt_history"] == durable_history
    assert persisted_image["selected_attempt"] == 3
    assert persisted_image["retry_policy"] == ("original+autocontrast-cutoff-1+otsu-max3-v2")


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

    assert retry_calls == 2
    assert (page_texts, chars) == ([""], 0)
    assert [item["source_page"] for item in page_errors] == [1]
    assert page_metadata[0]["ocr_outcome"] == "zero_output"
    assert page_metadata[0]["attempt_count"] == 3


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


def test_surya_third_retry_requires_mandatory_geometry_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_calls = 0

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        nonlocal retry_calls
        retry_calls += 1
        if retry_calls == 1:
            _write_surya_retry_sidecar(image_paths=image_paths, work_dir=work_dir, text="")
            return "", 0
        return "SOLD OUT", len("SOLD OUT")

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    with pytest.raises(RuntimeError, match="third zero-output retry.*mandatory geometry"):
        _run_surya_pagewise(tmp_path, defer_empty_pages=True)
    assert retry_calls == 2


@pytest.mark.parametrize(
    "pages",
    [
        [
            {
                "image_bbox": [0, 0, 120, 80],
                "text_lines": [{"text": "SOLD OUT"}],
            }
        ],
        [
            {
                "image_bbox": [0, 0, 120, 80],
                "text_lines": [{"text": "SOLD OUT", "bbox": [1, 2, float("nan"), 20]}],
            }
        ],
        [
            {
                "image_bbox": [0, 0, 120, 80],
                "text_lines": [{"text": "SOLD OUT", "bbox": [1, 2, 121, 20]}],
            }
        ],
        [
            {
                "image_bbox": [0, 0, 120, 80],
                "text_lines": [{"text": "SOLD OUT", "bbox": [1, 2, 50, 20]}],
            },
            {"image_bbox": [0, 0, 120, 80], "text_lines": []},
        ],
    ],
    ids=["text-without-bbox", "non-finite-bbox", "out-of-bounds-bbox", "extra-page"],
)
def test_surya_third_retry_rejects_malformed_geometry(
    tmp_path: Path,
    monkeypatch,
    pages: list[dict[str, object]],
) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_calls = 0

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        nonlocal retry_calls
        retry_calls += 1
        if retry_calls == 1:
            _write_surya_retry_sidecar(image_paths=image_paths, work_dir=work_dir, text="")
            return "", 0
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "image_name": image_paths[0].name,
                            "pages": pages,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return "SOLD OUT", len("SOLD OUT")

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    expected_error = "Surya sidecar" if len(pages) != 1 else "third-attempt geometry"
    with pytest.raises(RuntimeError, match=expected_error):
        _run_surya_pagewise(tmp_path, defer_empty_pages=True)
    assert retry_calls == 2


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate-image",
        "wrong-image-name",
        "missing-image-bbox",
        "bool-image-bbox",
        "string-image-bbox",
        "infinite-image-bbox",
        "negative-image-bbox",
        "zero-image-bbox",
        "missing-line-bbox",
        "bool-line-bbox",
        "string-line-bbox",
        "negative-line-bbox",
        "reversed-line-bbox",
        "zero-line-bbox",
        "empty-line-text",
        "non-string-line-text",
        "non-object-line",
        "wrong-execution-path",
    ],
)
def test_surya_third_retry_strict_geometry_matrix(
    tmp_path: Path,
    monkeypatch,
    defect: str,
) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_calls = 0

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        nonlocal retry_calls
        retry_calls += 1
        if retry_calls == 1:
            _write_surya_retry_sidecar(image_paths=image_paths, work_dir=work_dir, text="")
            return "", 0

        image_name = image_paths[0].name
        text_line: Any = {"text": "SOLD OUT", "bbox": [1, 2, 50, 20]}
        page: dict[str, Any] = {
            "image_bbox": [0, 0, 120, 80],
            "text_lines": [text_line],
        }
        image: dict[str, Any] = {"image_name": image_name, "pages": [page]}
        images: list[dict[str, Any]] = [image]
        payload: dict[str, Any] = {"execution_path": "module", "images": images}
        if defect == "duplicate-image":
            images.append(dict(image))
        elif defect == "wrong-image-name":
            image["image_name"] = "other.png"
        elif defect == "missing-image-bbox":
            del page["image_bbox"]
        elif defect == "bool-image-bbox":
            page["image_bbox"] = [False, 0, 120, 80]
        elif defect == "string-image-bbox":
            page["image_bbox"] = ["0", 0, 120, 80]
        elif defect == "infinite-image-bbox":
            page["image_bbox"] = [0, 0, float("inf"), 80]
        elif defect == "negative-image-bbox":
            page["image_bbox"] = [-1, 0, 120, 80]
        elif defect == "zero-image-bbox":
            page["image_bbox"] = [0, 0, 0, 80]
        elif defect == "missing-line-bbox":
            del text_line["bbox"]
        elif defect == "bool-line-bbox":
            text_line["bbox"] = [True, 2, 50, 20]
        elif defect == "string-line-bbox":
            text_line["bbox"] = ["1", 2, 50, 20]
        elif defect == "negative-line-bbox":
            text_line["bbox"] = [-1, 2, 50, 20]
        elif defect == "reversed-line-bbox":
            text_line["bbox"] = [50, 2, 1, 20]
        elif defect == "zero-line-bbox":
            text_line["bbox"] = [1, 2, 1, 20]
        elif defect == "empty-line-text":
            text_line["text"] = "---"
        elif defect == "non-string-line-text":
            text_line["text"] = 42
        elif defect == "non-object-line":
            page["text_lines"] = ["SOLD OUT"]
        else:
            payload["execution_path"] = "fallback"
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        return "SOLD OUT", len("SOLD OUT")

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    expected_error = (
        "Surya sidecar"
        if defect in {"duplicate-image", "wrong-image-name"}
        else "third-attempt geometry"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        _run_surya_pagewise(tmp_path, defer_empty_pages=True)
    assert retry_calls == 2


_RAW_SIDECAR_DEFECTS = (
    "duplicate-text-zero",
    "duplicate-zero-text",
    "duplicate-zero",
    "nonobject",
    "unexpected-extra",
    "extra-page",
    "malformed-page",
)


@pytest.mark.parametrize("defect", _RAW_SIDECAR_DEFECTS)
def test_surya_malformed_raw_initial_sidecar_never_starts_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)

    def fake_direct(image_paths, *, lang, work_dir, which_fn, run_cmd):
        _write_malformed_raw_surya_sidecar(
            image_path=image_paths[0],
            work_dir=work_dir,
            defect=defect,
        )
        return "", 0

    retry_calls = 0

    def retry(*_args, **_kwargs):
        nonlocal retry_calls
        retry_calls += 1
        raise AssertionError("malformed raw attempt 1 must not be retried")

    monkeypatch.setattr(benchmark, "_run_surya_direct", fake_direct)
    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    with pytest.raises(RuntimeError, match="Surya sidecar"):
        _run_surya_pagewise(tmp_path, defer_empty_pages=True)
    assert retry_calls == 0
    assert not (tmp_path / "work" / "zero_output_retry").exists()


@pytest.mark.parametrize("defect", _RAW_SIDECAR_DEFECTS)
def test_surya_malformed_raw_second_sidecar_never_starts_otsu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_calls = 0

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        nonlocal retry_calls
        retry_calls += 1
        if retry_calls > 1:
            raise AssertionError("malformed raw attempt 2 must not start Otsu")
        _write_malformed_raw_surya_sidecar(
            image_path=image_paths[0],
            work_dir=work_dir,
            defect=defect,
        )
        return "", 0

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    with pytest.raises(RuntimeError, match="Surya sidecar"):
        _run_surya_pagewise(tmp_path, defer_empty_pages=True)
    assert retry_calls == 1
    assert not (tmp_path / "work" / "zero_output_retry" / "page_0001" / "attempt_3_otsu").exists()


@pytest.mark.parametrize(
    "defect",
    ["extra-page", "bool-image-bbox", "wrong-execution-path"],
)
def test_surya_malformed_second_zero_output_never_reaches_third_attempt(
    tmp_path: Path,
    monkeypatch,
    defect: str,
) -> None:
    _install_initial_surya_zero(monkeypatch)
    retry_calls = 0

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        nonlocal retry_calls
        retry_calls += 1
        _write_surya_retry_sidecar(image_paths=image_paths, work_dir=work_dir, text="")
        sidecar = work_dir / "surya_page_lines.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if defect == "extra-page":
            payload["images"][0]["pages"].append({"image_bbox": [0, 0, 120, 80], "text_lines": []})
        elif defect == "bool-image-bbox":
            payload["images"][0]["pages"][0]["image_bbox"] = [False, 0, 120, 80]
        else:
            payload["execution_path"] = "fallback"
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        return "", 0

    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    expected_error = "Surya sidecar" if defect == "extra-page" else "second zero-output evidence"
    with pytest.raises(RuntimeError, match=expected_error):
        _run_surya_pagewise(tmp_path, defer_empty_pages=True)
    assert retry_calls == 1


@pytest.mark.parametrize(
    "defect",
    ["extra-page", "bool-image-bbox", "wrong-execution-path"],
)
def test_surya_malformed_initial_zero_output_never_starts_retry(
    tmp_path: Path,
    monkeypatch,
    defect: str,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)

    def fake_direct(image_paths, *, lang, work_dir, which_fn, run_cmd):
        page = {"image_bbox": [0, 0, 120, 80], "text_lines": []}
        pages = [page]
        payload = {
            "execution_path": "module",
            "images": [{"image_name": image_paths[0].name, "pages": pages}],
        }
        if defect == "extra-page":
            pages.append({"image_bbox": [0, 0, 120, 80], "text_lines": []})
        elif defect == "bool-image-bbox":
            page["image_bbox"] = [False, 0, 120, 80]
        else:
            payload["execution_path"] = "fallback"
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        return "", 0

    retry_calls = 0

    def retry(*_args, **_kwargs):
        nonlocal retry_calls
        retry_calls += 1
        raise AssertionError("malformed attempt 1 must not be retried")

    monkeypatch.setattr(benchmark, "_run_surya_direct", fake_direct)
    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    expected_error = "Surya sidecar" if defect == "extra-page" else "initial zero-output evidence"
    with pytest.raises(RuntimeError, match=expected_error):
        benchmark._run_extraction_engine_pagewise(
            OCR_ENGINE_SURYA,
            [image_path],
            source_pages_1based=[1],
            lang="eng",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
            defer_empty_pages=True,
        )
    assert retry_calls == 0


def test_surya_wrong_initial_image_name_is_not_zero_output_retry_eligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)

    def fake_direct(image_paths, *, lang, work_dir, which_fn, run_cmd):
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "execution_path": "module",
                    "images": [
                        {
                            "image_name": "other.png",
                            "pages": [{"image_bbox": [0, 0, 120, 80], "text_lines": []}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return "", 0

    monkeypatch.setattr(benchmark, "_run_surya_direct", fake_direct)
    monkeypatch.setattr(
        benchmark,
        "_run_surya_module_cli",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected raw image must not be retried")
        ),
    )

    with pytest.raises(RuntimeError, match="Surya sidecar image names"):
        benchmark._run_extraction_engine_pagewise(
            OCR_ENGINE_SURYA,
            [image_path],
            source_pages_1based=[1],
            lang="eng",
            work_dir=tmp_path / "work",
            which_fn=lambda _name: None,
            run_cmd=lambda *_args, **_kwargs: None,
            defer_empty_pages=True,
        )


def test_surya_verified_blank_never_starts_zero_output_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (120, 80), "white").save(image_path)

    def fake_direct(image_paths, *, lang, work_dir, which_fn, run_cmd):
        _write_surya_retry_sidecar(image_paths=image_paths, work_dir=work_dir, text="")
        return "", 0

    monkeypatch.setattr(benchmark, "_run_surya_direct", fake_direct)
    monkeypatch.setattr(
        benchmark,
        "_run_surya_module_cli",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("verified blank must not be retried")
        ),
    )

    page_texts, _, page_errors, page_metadata = benchmark._run_extraction_engine_pagewise(
        OCR_ENGINE_SURYA,
        [image_path],
        source_pages_1based=[1],
        lang="eng",
        work_dir=tmp_path / "work",
        which_fn=lambda _name: None,
        run_cmd=lambda *_args, **_kwargs: None,
        defer_empty_pages=True,
    )

    assert page_texts == [""]
    assert page_errors == []
    assert page_metadata[0]["ocr_outcome"] == "verified_blank"
    assert page_metadata[0]["attempt_count"] == 1


def test_surya_third_retry_is_scoped_to_initial_zero_output_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for image_path in image_paths:
        _nonblank_image(image_path)

    def fake_direct(image_paths, *, lang, work_dir, which_fn, run_cmd):
        sidecar = work_dir / "surya_page_lines.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "execution_path": "module",
                    "images": [
                        {
                            "image_name": image_paths[0].name,
                            "pages": [
                                {
                                    "image_bbox": [0, 0, 120, 80],
                                    "text_lines": [{"text": "READY", "bbox": [1, 2, 50, 20]}],
                                }
                            ],
                        },
                        {
                            "image_name": image_paths[1].name,
                            "pages": [{"image_bbox": [0, 0, 120, 80], "text_lines": []}],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return "READY", len("READY")

    retried_names: list[str] = []

    def retry(image_paths, *, lang, work_dir, which_fn, run_cmd):
        retried_names.append(image_paths[0].name)
        text = "" if len(retried_names) == 1 else "SOLD OUT"
        _write_surya_retry_sidecar(image_paths=image_paths, work_dir=work_dir, text=text)
        return text, len(text)

    monkeypatch.setattr(benchmark, "_run_surya_direct", fake_direct)
    monkeypatch.setattr(benchmark, "_run_surya_module_cli", retry)

    page_texts, _, page_errors, page_metadata = benchmark._run_extraction_engine_pagewise(
        OCR_ENGINE_SURYA,
        image_paths,
        source_pages_1based=[1, 2],
        lang="eng",
        work_dir=tmp_path / "work",
        which_fn=lambda _name: None,
        run_cmd=lambda *_args, **_kwargs: None,
        defer_empty_pages=True,
    )

    assert retried_names == ["second.png", "second.png"]
    assert page_texts == ["READY", "SOLD OUT"]
    assert page_errors == []
    assert {row["source_page"]: row["attempt_count"] for row in page_metadata} == {1: 1, 2: 3}


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
    assert retry_calls == 2
