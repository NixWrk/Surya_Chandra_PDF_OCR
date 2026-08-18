from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest
from PIL import Image

from uniscan.app import ocr_pipeline
import uniscan.ocr.benchmark as benchmark
from uniscan.ocr import OCR_ENGINE_SURYA


def _nonblank_image(path: Path) -> None:
    image = Image.new("RGB", (120, 80), (80, 80, 80))
    for x in range(20, 100):
        for y in range(20, 60):
            image.putpixel((x, y), (120, 120, 120))
    image.save(path)


def test_copy_retry_evidence_is_bounded_and_exclusive(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"seal")
    target = tmp_path / "owned" / "copy.bin"

    assert (
        benchmark._copy_retry_evidence(
            source=source,
            target=target,
            max_bytes=4,
        ).read_bytes()
        == b"seal"
    )
    with pytest.raises(RuntimeError, match="already exists"):
        benchmark._copy_retry_evidence(source=source, target=target, max_bytes=4)

    oversized = tmp_path / "oversized.bin"
    oversized.write_bytes(b"12345")
    with pytest.raises(RuntimeError, match="exceeds byte limit"):
        benchmark._copy_retry_evidence(
            source=oversized,
            target=tmp_path / "oversized-copy.bin",
            max_bytes=4,
        )


def test_copy_retry_evidence_removes_invalid_final_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "owned" / "copy.bin"
    source.write_bytes(b"seal")
    real_fingerprint = benchmark._stable_file_fingerprint
    target_reads = 0

    def mismatched_final_fingerprint(path: Path) -> dict[str, object]:
        nonlocal target_reads
        fingerprint = real_fingerprint(path)
        if path == target:
            target_reads += 1
            if target_reads == 2:
                return {"sha256": "0" * 64, "bytes": fingerprint["bytes"]}
        return fingerprint

    monkeypatch.setattr(benchmark, "_stable_file_fingerprint", mismatched_final_fingerprint)
    with pytest.raises(RuntimeError, match="target seal is invalid"):
        benchmark._copy_retry_evidence(source=source, target=target, max_bytes=4)
    assert target_reads == 2
    assert not target.exists()


def test_copy_retry_evidence_rejects_linked_source(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    linked = tmp_path / "linked.bin"
    source.write_bytes(b"seal")
    os.link(source, linked)

    with pytest.raises(RuntimeError, match="not singly owned"):
        benchmark._copy_retry_evidence(
            source=linked,
            target=tmp_path / "copy.bin",
            max_bytes=4,
        )


def test_copy_retry_evidence_rejects_linked_target_parent(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"seal")
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        os.symlink(real_parent, linked_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")

    with pytest.raises(RuntimeError, match="target parent is unsafe"):
        benchmark._copy_retry_evidence(
            source=source,
            target=linked_parent / "nested" / "copy.bin",
            max_bytes=4,
        )
    assert not (real_parent / "nested").exists()
    assert not (real_parent / "nested" / "copy.bin").exists()


def _install_fake_chandra(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generate: Callable[..., list[Any]],
    load_image: Callable[..., Image.Image] | None = None,
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
    prompts_module = ModuleType("chandra.prompts")
    prompts_module.PROMPT_MAPPING = {
        "ocr_layout": "fake layout prompt",
        "ocr": "fake plain prompt",
    }
    input_module = ModuleType("chandra.input")

    def fake_load_image(path: str, min_image_dim: int = 1536) -> Image.Image:
        image = Image.open(path).convert("RGB")
        if image.width < min_image_dim or image.height < min_image_dim:
            scale = min_image_dim / float(min(image.width, image.height))
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        return image

    input_module.load_image = load_image or fake_load_image
    chandra_module = ModuleType("chandra")
    chandra_module.model = model_module
    monkeypatch.setitem(sys.modules, "chandra", chandra_module)
    monkeypatch.setitem(sys.modules, "chandra.model", model_module)
    monkeypatch.setitem(sys.modules, "chandra.model.schema", schema_module)
    monkeypatch.setitem(sys.modules, "chandra.prompts", prompts_module)
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
    generated_prompts: list[str] = []
    autocontrast_calls: list[tuple[tuple[int, int], int]] = []
    real_autocontrast = benchmark.ImageOps.autocontrast

    def spy_autocontrast(image, cutoff=0, **kwargs):
        autocontrast_calls.append((image.size, cutoff))
        return real_autocontrast(image, cutoff=cutoff, **kwargs)

    def generate(batch):
        generated_sizes.append(batch[0]["image"].size)
        generated_prompts.append(batch[0]["prompt_type"])
        if len(generated_sizes) == 1:
            return [SimpleNamespace(chunks=[], markdown="")]
        return [
            SimpleNamespace(
                chunks=[{"label": "Text", "content": "RECOVERED", "bbox": [1, 2, 50, 20]}],
                markdown="",
            )
        ]

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 123_456)
    _install_fake_chandra(monkeypatch, generate=generate)
    monkeypatch.setattr(benchmark.ImageOps, "autocontrast", spy_autocontrast)

    text, chars = benchmark._run_chandra_module(
        [image_path],
        lang="eng",
        work_dir=tmp_path / "work",
    )

    assert text == "RECOVERED"
    assert chars == len("RECOVERED")
    assert generated_sizes == [(2304, 1536), (2304, 1536)]
    assert generated_prompts == ["ocr_layout", "ocr_layout"]
    assert autocontrast_calls == [((2304, 1536), 1)]
    sidecar = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )
    image_evidence = sidecar["images"][0]
    source_identity = image_evidence["source_raster_identity"]
    assert source_identity == image_evidence["attempts"][0]["source_raster_identity"]
    assert source_identity["name"] == image_path.name
    assert source_identity["source_page"] == 1
    assert source_identity["verified_blank"] is False
    assert len(source_identity["pixel_sha256"]) == 64
    assert image_evidence["ocr_outcome"] == "text"
    assert image_evidence["attempt_count"] == 2
    assert image_evidence["selected_attempt"] == 2
    assert image_evidence["terminal_attempt"] == 2
    assert image_evidence["chandra_retry_policy"] == (
        "ocr-layout-original+ocr-layout-autocontrast-cutoff-1+ocr-original-max3-v1"
    )
    assert image_evidence["retry_preprocessing"] == "autocontrast-cutoff-1"
    assert [item["prompt_type"] for item in image_evidence["attempts"]] == [
        "ocr_layout",
        "ocr_layout",
    ]
    assert Image.MAX_IMAGE_PIXELS == 123_456


def test_chandra_retry_records_explicit_graphics_after_three_zero_text_attempts(
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
    assert calls == 3
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["ocr_outcome"] == "explicit_nontext"
    assert evidence["explicit_nontext"] is True
    assert evidence["chandra_non_text_labels"] == ["figure"]
    assert evidence["attempt_count"] == 3
    assert evidence["terminal_attempt"] == 3
    assert evidence["retry_preprocessing"] == "plain-ocr-original-v1"
    assert [item["prompt_type"] for item in evidence["attempts"]] == [
        "ocr_layout",
        "ocr_layout",
        "ocr",
    ]
    assert evidence["pages"][0]["text_lines"] == []


def test_chandra_preserves_original_label_but_stays_zero_without_layout_consensus(
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
    assert calls == 3
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["ocr_outcome"] == "zero_output"
    assert evidence["explicit_nontext"] is False
    assert evidence["chandra_non_text_labels"] == ["figure"]


def test_chandra_requires_explicit_graphic_evidence_from_all_three_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    calls = 0

    def generate(_batch):
        nonlocal calls
        calls += 1
        if calls < 3:
            return [
                SimpleNamespace(
                    chunks=[{"label": "Figure", "content": "", "bbox": [0, 0, 120, 80]}],
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
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["ocr_outcome"] == "zero_output"
    assert evidence["explicit_nontext"] is False
    assert [attempt["explicit_nontext"] for attempt in evidence["attempts"]] == [
        True,
        True,
        False,
    ]


def test_chandra_plain_third_attempt_recovers_text_inside_image_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    prompts: list[str] = []

    def generate(batch):
        prompt_type = batch[0]["prompt_type"]
        prompts.append(prompt_type)
        if prompt_type == "ocr_layout":
            return [
                SimpleNamespace(
                    chunks=[{"label": "Image", "content": "", "bbox": [0, 0, 120, 80]}],
                    markdown="",
                )
            ]
        return [
            SimpleNamespace(
                chunks=[
                    {
                        "label": "Image",
                        "content": (
                            '<img alt="English description must be ignored"/>'
                            "<div>Detailed visual description must be ignored</div>"
                            "<p>НАД ЧЕМ<br/>РАБОТАЕШЬ?</p>"
                        ),
                        "bbox": [10, 12, 110, 68],
                    }
                ],
                markdown="",
            )
        ]

    _install_fake_chandra(monkeypatch, generate=generate)

    text, chars = benchmark._run_chandra_module(
        [image_path],
        lang="rus+eng",
        work_dir=tmp_path / "work",
    )

    assert text == "НАД ЧЕМ\nРАБОТАЕШЬ?"
    assert chars == len(text)
    assert prompts == ["ocr_layout", "ocr_layout", "ocr"]
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["ocr_outcome"] == "text"
    assert evidence["selected_attempt"] == 3
    assert evidence["attempt_count"] == 3
    assert evidence["attempts"][2]["ocr_outcome"] == "text"
    assert evidence["attempts"][2]["geometry_lines"] == 2
    assert [line["text"] for line in evidence["pages"][0]["text_lines"]] == [
        "НАД ЧЕМ",
        "РАБОТАЕШЬ?",
    ]


def test_chandra_plain_third_attempt_rejects_text_without_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    calls = 0

    def generate(_batch):
        nonlocal calls
        calls += 1
        if calls < 3:
            return [SimpleNamespace(chunks=[], markdown="")]
        return [
            SimpleNamespace(
                chunks=[{"label": "Image", "content": "<p>UNSEALED</p>"}],
                markdown="",
            )
        ]

    _install_fake_chandra(monkeypatch, generate=generate)

    with pytest.raises(RuntimeError, match="OCR recovered text without complete geometry"):
        benchmark._run_chandra_module(
            [image_path],
            lang="eng",
            work_dir=tmp_path / "work",
        )
    assert calls == 3


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(chunks=[], markdown="UNSEALED"),
        SimpleNamespace(
            chunks=[
                {"label": "Text", "content": "SEALED", "bbox": [1, 2, 50, 20]},
                {"label": "Text", "content": "UNSEALED"},
            ],
            markdown="",
        ),
    ],
)
def test_chandra_layout_rejects_text_without_complete_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: SimpleNamespace,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)

    _install_fake_chandra(monkeypatch, generate=lambda _batch: [result])

    with pytest.raises(RuntimeError, match="OCR recovered text without complete geometry"):
        benchmark._run_chandra_module(
            [image_path],
            lang="eng",
            work_dir=tmp_path / "work",
        )


def test_chandra_plain_description_only_is_not_accepted_as_text(
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
                chunks=[
                    {
                        "label": "Image",
                        "content": (
                            '<img alt="invented alt text"/>'
                            "<div>Detailed visual description only</div>"
                        ),
                        "bbox": [0, 0, 120, 80],
                    }
                ],
                markdown="Detailed visual description only",
            )
        ]

    _install_fake_chandra(monkeypatch, generate=generate)

    text, chars = benchmark._run_chandra_module(
        [image_path],
        lang="eng",
        work_dir=tmp_path / "work",
    )

    assert (text, chars) == ("", 0)
    assert calls == 3
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["ocr_outcome"] == "explicit_nontext"
    assert all(attempt["text_chars"] == 0 for attempt in evidence["attempts"])
    assert all(
        attempt["alternative_text_evidence"]["accounting"] == "ignored_graphic_description"
        for attempt in evidence["attempts"]
    )


@pytest.mark.parametrize(
    ("alternative_field", "alternative_value"),
    [
        ("html", "<p>VISIBLE DIALOGUE</p>"),
        ("markdown", "VISIBLE DIALOGUE"),
    ],
)
def test_chandra_unaccounted_alternative_text_cannot_be_explicit_nontext(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alternative_field: str,
    alternative_value: str,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    calls = 0

    def generate(_batch):
        nonlocal calls
        calls += 1
        payload = {
            "chunks": [
                {
                    "label": "Image",
                    "content": "<div>Detailed visual description only</div>",
                    "bbox": [0, 0, 120, 80],
                }
            ],
            "html": "",
            "markdown": "",
        }
        payload[alternative_field] = alternative_value
        return [SimpleNamespace(**payload)]

    _install_fake_chandra(monkeypatch, generate=generate)

    text, chars = benchmark._run_chandra_module(
        [image_path],
        lang="eng",
        work_dir=tmp_path / "work",
    )

    assert (text, chars) == ("", 0)
    assert calls == 3
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["ocr_outcome"] == "zero_output"
    assert evidence["explicit_nontext"] is False
    assert all(attempt["explicit_nontext"] is False for attempt in evidence["attempts"])
    assert all(
        attempt["alternative_text_evidence"]["accounting"] == "unaccounted"
        for attempt in evidence["attempts"]
    )


@pytest.mark.parametrize("prompt_mode", ["layout", "plain"])
@pytest.mark.parametrize(
    ("alternative_field", "alternative_value"),
    [
        ("html", "<p>CONFLICTING ALTERNATIVE</p>"),
        ("markdown", "CONFLICTING ALTERNATIVE"),
    ],
)
def test_chandra_text_attempt_rejects_unaccounted_alternative_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_mode: str,
    alternative_field: str,
    alternative_value: str,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    calls = 0

    def generate(_batch):
        nonlocal calls
        calls += 1
        if prompt_mode == "plain" and calls < 3:
            return [SimpleNamespace(chunks=[], html="", markdown="")]
        content = "VISIBLE TEXT" if prompt_mode == "layout" else "<p>VISIBLE TEXT</p>"
        payload = {
            "chunks": [
                {
                    "label": "Text" if prompt_mode == "layout" else "Image",
                    "content": content,
                    "bbox": [0, 0, 120, 80],
                }
            ],
            "html": "",
            "markdown": "",
        }
        payload[alternative_field] = alternative_value
        return [SimpleNamespace(**payload)]

    _install_fake_chandra(monkeypatch, generate=generate)

    with pytest.raises(RuntimeError, match="unaccounted alternative text"):
        benchmark._run_chandra_module(
            [image_path],
            lang="eng",
            work_dir=tmp_path / "work",
        )
    assert calls == (1 if prompt_mode == "layout" else 3)


def test_chandra_layout_accepts_intentionally_omitted_header_footer(
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
                chunks=[
                    {
                        "label": "Page-Header",
                        "content": "DOCUMENT HEADER",
                        "bbox": [0, 0, 120, 20],
                    },
                    {
                        "label": "Text",
                        "content": "VISIBLE BODY",
                        "bbox": [0, 20, 120, 60],
                    },
                    {
                        "label": "Page-Footer",
                        "content": "PAGE 3",
                        "bbox": [0, 60, 120, 80],
                    },
                ],
                html="<p>VISIBLE BODY</p>",
                markdown="VISIBLE BODY",
            )
        ]

    _install_fake_chandra(monkeypatch, generate=generate)

    text, chars = benchmark._run_chandra_module(
        [image_path],
        lang="eng",
        work_dir=tmp_path / "work",
    )

    assert text == "DOCUMENT HEADER\nVISIBLE BODY\nPAGE 3"
    assert chars == len(text)
    assert calls == 1
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["attempts"][0]["alternative_text_evidence"]["accounting"] == (
        "parsed_without_header_footer"
    )


def test_chandra_layout_accepts_markdown_list_marker_loss_after_header_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    body = (
        "1. This numbered paragraph is deliberately long enough for strict near-complete "
        "coverage after Markdown removes its list marker."
    )

    def generate(_batch):
        return [
            SimpleNamespace(
                chunks=[
                    {"label": "Page-Header", "content": "PAGE 3", "bbox": [0, 0, 120, 20]},
                    {"label": "Text", "content": body, "bbox": [0, 20, 120, 80]},
                ],
                html=f"<p>{body}</p>",
                markdown=body,
            )
        ]

    _install_fake_chandra(monkeypatch, generate=generate)

    text, chars = benchmark._run_chandra_module(
        [image_path],
        lang="eng",
        work_dir=tmp_path / "work",
    )

    assert text == f"PAGE 3\n{body}"
    assert chars == len(text)
    evidence = json.loads(
        (tmp_path / "work" / "chandra_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert evidence["attempts"][0]["alternative_text_evidence"]["accounting"] == (
        "parsed_without_header_footer"
    )


def test_chandra_layout_rejects_conflicting_markdown_despite_matching_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    body = "VISIBLE BODY TEXT WITH COMPLETE GEOMETRY"

    def generate(_batch):
        return [
            SimpleNamespace(
                chunks=[
                    {"label": "Page-Header", "content": "PAGE 3", "bbox": [0, 0, 120, 20]},
                    {"label": "Text", "content": body, "bbox": [0, 20, 120, 80]},
                ],
                html=f"<p>{body}</p>",
                markdown=f"{body}\nALIEN ALTERNATIVE TEXT",
            )
        ]

    _install_fake_chandra(monkeypatch, generate=generate)

    with pytest.raises(RuntimeError, match="unaccounted alternative text"):
        benchmark._run_chandra_module(
            [image_path],
            lang="eng",
            work_dir=tmp_path / "work",
        )


@pytest.mark.parametrize("defect", ["empty", "extra", "error"])
def test_chandra_attempt_rejects_invalid_batch_cardinality_or_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)

    def generate(_batch):
        result = SimpleNamespace(chunks=[], markdown="", error=defect == "error")
        if defect == "empty":
            return []
        if defect == "extra":
            return [result, result]
        return [result]

    _install_fake_chandra(monkeypatch, generate=generate)

    with pytest.raises(RuntimeError, match="returned|marked"):
        benchmark._run_chandra_module(
            [image_path],
            lang="eng",
            work_dir=tmp_path / "work",
        )


def test_chandra_attempt_rejects_error_from_mapping_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    _install_fake_chandra(
        monkeypatch,
        generate=lambda _batch: [{"chunks": [], "html": "", "markdown": "", "error": True}],
    )

    with pytest.raises(RuntimeError, match="marked"):
        benchmark._run_chandra_module(
            [image_path],
            lang="eng",
            work_dir=tmp_path / "work",
        )


def test_chandra_exception_does_not_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    calls = 0

    def generate(_batch):
        nonlocal calls
        calls += 1
        raise MemoryError("oom")

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 123_456)
    _install_fake_chandra(monkeypatch, generate=generate)

    with pytest.raises(MemoryError, match="oom"):
        benchmark._run_chandra_module(
            [image_path],
            lang="eng",
            work_dir=tmp_path / "work",
        )
    assert calls == 1
    assert Image.MAX_IMAGE_PIXELS == 123_456


def test_chandra_rejects_extreme_aspect_before_model_image_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "extreme.png"
    Image.new("RGB", (1, 32_768), "white").save(image_path)
    load_calls: list[str] = []

    def forbidden_load(path: str, *, min_image_dim: int) -> Image.Image:
        load_calls.append(path)
        raise AssertionError(f"load_image must not be called (min_image_dim={min_image_dim})")

    _install_fake_chandra(
        monkeypatch,
        generate=lambda _batch: (_ for _ in ()).throw(
            AssertionError("model.generate must not be called")
        ),
        load_image=forbidden_load,
    )

    with pytest.raises(RuntimeError, match="model input would exceed"):
        benchmark._run_chandra_module(
            [image_path],
            lang="eng",
            work_dir=tmp_path / "work",
        )
    assert load_calls == []


def test_chandra_rejects_unexpected_loader_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "page.png"
    _nonblank_image(image_path)
    _install_fake_chandra(
        monkeypatch,
        generate=lambda _batch: [],
        load_image=lambda *_args, **_kwargs: Image.new("RGB", (1536, 1536), "white"),
    )

    with pytest.raises(RuntimeError, match="unexpected model input dimensions"):
        benchmark._run_chandra_module([image_path], lang="eng", work_dir=tmp_path / "work")


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

    def fake_direct(
        image_paths,
        *,
        lang,
        work_dir,
        which_fn,
        run_cmd,
        page_progress_cb,
        source_raster_identities,
    ):
        assert len(source_raster_identities) == len(image_paths)
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
    bbox = [35, 25, 55, 35] if "attempt_3_scaled" in retry_image.parts else [1, 2, 50, 20]
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
                                "text_lines": ([{"text": text, "bbox": bbox}] if text else []),
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


def _persist_and_validate_surya_page(
    tmp_path: Path,
    *,
    page_texts: list[str],
    page_errors: list[dict[str, Any]],
    page_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = tmp_path / "validated-output"
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
    engine_dir = output_dir / OCR_ENGINE_SURYA
    pages_payload = json.loads((engine_dir / "pages.json").read_text(encoding="utf-8"))
    row = pages_payload["pages"][0]
    attempt, _source_identity = ocr_pipeline._strict_surya_attempt_metadata(
        row=row,
        engine_dir=engine_dir,
        source_page=1,
    )
    assert attempt == row["attempt_count"]
    return row


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
    retry_attempt = page_metadata[0]["attempt_history"][1]
    retry_sidecar_path = Path(retry_attempt["sidecar_path"])
    retry_sidecar = json.loads(retry_sidecar_path.read_text(encoding="utf-8"))
    retry_image = retry_sidecar["images"][0]
    assert retry_image["ocr_outcome"] == "text"
    assert retry_image["attempt_count"] == 2
    assert retry_image["retry_preprocessing"] == "autocontrast-cutoff-1"
    assert (
        hashlib.sha256(retry_sidecar_path.read_bytes()).hexdigest()
        == (retry_attempt["sidecar_sha256"])
    )
    raw_initial = json.loads(
        (tmp_path / "work" / "batch" / "surya_page_lines.json").read_text(encoding="utf-8")
    )["images"][0]
    assert "ocr_outcome" not in raw_initial
    assert "attempt_count" not in raw_initial
    assert "retry_preprocessing" not in raw_initial
    validated = _persist_and_validate_surya_page(
        tmp_path,
        page_texts=page_texts,
        page_errors=page_errors,
        page_metadata=page_metadata,
    )
    assert validated["attempt_count"] == 2


def test_surya_third_retry_scales_content_and_recovers(
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
            assert image.mode == "RGB"
            assert image.getpixel((0, 0)) == (255, 255, 255)
            assert image.getpixel((60, 40)) == (120, 120, 120)
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
    assert page_metadata[0]["retry_preprocessing"] == ("rgb-scale-0.5-center-white-lanczos-v1")
    assert autocontrast_calls == [("RGB", (120, 80), 1)]
    assert page_metadata[0]["selected_attempt"] == 3
    assert page_metadata[0]["retry_policy"] == (
        "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
    )
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
        "rgb-scale-0.5-center-white-lanczos-v1",
    ]
    assert attempts[2]["image_size"] == [120, 80]
    assert attempts[2]["content_scale"] == 0.5
    assert attempts[2]["content_size"] == [60, 40]
    assert attempts[2]["content_offset"] == [30, 20]
    assert attempts[2]["resampling"] == "lanczos"
    assert attempts[2]["canvas_fill_rgb"] == [255, 255, 255]
    assert len(attempts[2]["image_sha256"]) == 64
    assert all(len(item["sidecar_sha256"]) == 64 for item in attempts)
    durable = json.loads(
        Path(page_metadata[0]["surya_page_lines_path"]).read_text(encoding="utf-8")
    )["images"][0]
    assert durable["attempt_history"] == attempts
    assert durable["selected_attempt"] == 3
    assert durable["retry_policy"] == page_metadata[0]["retry_policy"]
    assert durable["geometry_coordinate_space"] == "source-image-v1"
    assert durable["geometry_transform"] == "inverse-actual-content-size-strict-v1"
    assert durable["pages"][0]["text_lines"][0]["bbox"] == [10.0, 10.0, 50.0, 30.0]
    raw_attempt = json.loads(Path(attempts[2]["sidecar_path"]).read_text(encoding="utf-8"))
    second_attempt = json.loads(Path(attempts[1]["sidecar_path"]).read_text(encoding="utf-8"))[
        "images"
    ][0]
    assert second_attempt["ocr_outcome"] == "zero_output"
    assert second_attempt["attempt_count"] == 2
    assert second_attempt["retry_preprocessing"] == "autocontrast-cutoff-1"
    raw_attempt_image = raw_attempt["images"][0]
    assert raw_attempt_image["ocr_outcome"] == "text"
    assert raw_attempt_image["attempt_count"] == 3
    assert raw_attempt_image["retry_preprocessing"] == ("rgb-scale-0.5-center-white-lanczos-v1")
    assert raw_attempt["images"][0]["pages"][0]["text_lines"][0]["bbox"] == [
        35,
        25,
        55,
        35,
    ]

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
    assert pages_payload["pages"][0]["geometry_coordinate_space"] == "source-image-v1"
    assert pages_payload["pages"][0]["geometry_transform"] == (
        "inverse-actual-content-size-strict-v1"
    )
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
    assert persisted_image["retry_policy"] == (
        "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
    )
    assert persisted_image["geometry_coordinate_space"] == "source-image-v1"
    assert persisted_image["geometry_transform"] == "inverse-actual-content-size-strict-v1"
    assert persisted_image["pages"][0]["text_lines"][0]["bbox"] == [10.0, 10.0, 50.0, 30.0]
    attempt, _source_identity = ocr_pipeline._strict_surya_attempt_metadata(
        row=pages_payload["pages"][0],
        engine_dir=output_dir / OCR_ENGINE_SURYA,
        source_page=1,
    )
    assert attempt == 3


def test_scaled_retry_uses_exact_odd_content_size_and_asymmetric_borders(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "scaled.png"
    Image.new("RGB", (1301, 1313), (120, 120, 120)).save(source, format="PNG")

    evidence = benchmark._write_scaled_retry_image(source=source, target=target)

    assert evidence["image_size"] == [1301, 1313]
    assert evidence["content_size"] == [650, 656]
    assert evidence["content_offset"] == [325, 328]
    with Image.open(target) as image:
        assert image.mode == "RGB"
        assert image.size == (1301, 1313)
        assert image.getpixel((324, 500)) == (255, 255, 255)
        assert image.getpixel((325, 500)) == (120, 120, 120)
        assert image.getpixel((974, 500)) == (120, 120, 120)
        assert image.getpixel((975, 500)) == (255, 255, 255)
        assert image.getpixel((500, 327)) == (255, 255, 255)
        assert image.getpixel((500, 328)) == (120, 120, 120)
        assert image.getpixel((500, 983)) == (120, 120, 120)
        assert image.getpixel((500, 984)) == (255, 255, 255)


def test_inverse_scaled_retry_bbox_uses_actual_odd_axis_scales() -> None:
    kwargs = {
        "source_size": [1301, 1313],
        "content_size": [650, 656],
        "content_offset": [325, 328],
        "label": "diagnostic bbox",
    }
    sold = benchmark._inverse_scaled_retry_bbox([351, 366, 938, 597], **kwargs)
    out = benchmark._inverse_scaled_retry_bbox([468, 675, 867, 870], **kwargs)

    assert sold == pytest.approx([52.04, 76.0579268292683, 1226.943076923077, 538.4100609756098])
    assert out == pytest.approx([286.22, 694.5289634146342, 1084.833846153846, 1084.8262195121952])
    assert benchmark._inverse_scaled_retry_bbox([325, 328, 975, 984], **kwargs) == pytest.approx(
        [0.0, 0.0, 1301.0, 1313.0]
    )


@pytest.mark.parametrize(
    "bbox",
    (
        [0, 0, 100, 100],
        [300, 300, 400, 400],
        [351, 366, 351, 597],
        [351, 366, float("nan"), 597],
    ),
)
def test_inverse_scaled_retry_bbox_rejects_unusable_geometry(bbox: list[float]) -> None:
    with pytest.raises(RuntimeError):
        benchmark._inverse_scaled_retry_bbox(
            bbox,
            source_size=[1301, 1313],
            content_size=[650, 656],
            content_offset=[325, 328],
            label="bad bbox",
        )


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
    attempts = page_metadata[0]["attempt_history"]
    for attempt, expected_count, expected_preprocessing in zip(
        attempts[1:],
        (2, 3),
        (
            "autocontrast-cutoff-1",
            "rgb-scale-0.5-center-white-lanczos-v1",
        ),
        strict=True,
    ):
        sidecar_path = Path(attempt["sidecar_path"])
        image = json.loads(sidecar_path.read_text(encoding="utf-8"))["images"][0]
        assert image["ocr_outcome"] == "zero_output"
        assert image["attempt_count"] == expected_count
        assert image["retry_preprocessing"] == expected_preprocessing
        assert hashlib.sha256(sidecar_path.read_bytes()).hexdigest() == (attempt["sidecar_sha256"])
    validated = _persist_and_validate_surya_page(
        tmp_path,
        page_texts=page_texts,
        page_errors=page_errors,
        page_metadata=page_metadata,
    )
    assert validated["attempt_count"] == 3
    assert validated["ocr_outcome"] == "zero_output"


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
def test_surya_malformed_raw_second_sidecar_never_starts_scaled_retry(
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
            raise AssertionError("malformed raw attempt 2 must not start scaled retry")
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
    assert not (tmp_path / "work" / "zero_output_retry" / "page_0001" / "attempt_3_scaled").exists()


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
    raw_sidecar = (
        tmp_path
        / "work"
        / "zero_output_retry"
        / "page_0001"
        / "attempt_2_autocontrast"
        / "module"
        / "surya_page_lines.json"
    )
    raw_image = json.loads(raw_sidecar.read_text(encoding="utf-8"))["images"][0]
    assert "ocr_outcome" not in raw_image
    assert "attempt_count" not in raw_image
    assert "retry_preprocessing" not in raw_image


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
    assert page_metadata[0]["selected_attempt"] == 1
    assert page_metadata[0]["retry_policy"] == (
        "original+autocontrast-cutoff-1+rgb-scale-0.5-center-white-lanczos-max3-v3"
    )
    assert len(page_metadata[0]["attempt_history"]) == 1
    validated = _persist_and_validate_surya_page(
        tmp_path, page_texts=page_texts, page_errors=page_errors, page_metadata=page_metadata
    )
    assert validated["selected_attempt"] == 1


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
