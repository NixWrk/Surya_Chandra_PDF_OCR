"""Chandra OCR engine plugin for OCRmyPDF."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from ocrmypdf import OrientationConfidence, hookimpl
from ocrmypdf.exceptions import BadArgsError, MissingDependencyError
from ocrmypdf.pluginspec import OcrEngine

from .hocr import build_hocr

Image.MAX_IMAGE_PIXELS = None
log = logging.getLogger(__name__)

_RUNTIME_CACHE: dict[str, Any] = {}
_WS_RE = re.compile(r"\s+")


def _method_from_options(options) -> str:
    return str(getattr(options, "chandra_method", "hf") or "hf").strip().lower()


def _safe_confidence(value: Any) -> float:
    try:
        conf = float(value or 0.0)
    except Exception:
        return 0.0
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def _safe_bbox(raw_bbox: Any, width: int, height: int) -> list[int]:
    width = max(1, int(width))
    height = max(1, int(height))
    try:
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise ValueError("bbox must have 4 values")
        x0, y0, x1, y1 = (
            int(float(raw_bbox[0])),
            int(float(raw_bbox[1])),
            int(float(raw_bbox[2])),
            int(float(raw_bbox[3])),
        )
    except Exception:
        return [0, 0, width, height]

    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return [x0, y0, x1, y1]


def _normalize_text(value: str) -> str:
    return _WS_RE.sub(" ", value or "").strip()


def _extract_chunk_text(raw_content: Any) -> str:
    text = ""
    if raw_content is None:
        return text

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(str(raw_content), "html.parser")
        text = soup.get_text(" ", strip=True)
    except Exception:
        text = str(raw_content)
    return _normalize_text(text)


def _resolve_chandra_runtime(*, method: str) -> Any:
    cached = _RUNTIME_CACHE.get(method)
    if cached is not None:
        return cached

    from chandra.model import InferenceManager

    manager = InferenceManager(method=method)
    runtime = SimpleNamespace(manager=manager)
    _RUNTIME_CACHE[method] = runtime
    return runtime


def _chunk_to_payload(*, chunk: Any, width: int, height: int) -> dict[str, Any] | None:
    if not isinstance(chunk, dict):
        return None

    text = _extract_chunk_text(chunk.get("content"))
    if not text:
        return None

    bbox = _safe_bbox(chunk.get("bbox"), width, height)
    confidence = _safe_confidence(chunk.get("confidence", 0.0))
    return {
        "text": text,
        "bbox": bbox,
        "confidence": confidence,
        "words": [],
    }


def _predict_chandra_lines(input_file: Path, options) -> tuple[list[dict[str, Any]], str, int, int]:
    method = _method_from_options(options)
    runtime = _resolve_chandra_runtime(method=method)

    prompt_type = str(getattr(options, "chandra_prompt_type", "ocr_layout") or "ocr_layout").strip() or "ocr_layout"
    include_images = bool(getattr(options, "chandra_include_images", False))
    include_headers_footers = bool(getattr(options, "chandra_include_headers_footers", False))
    vllm_api_base = str(getattr(options, "chandra_vllm_api_base", "") or "").strip()

    from chandra.model.schema import BatchInputItem

    with Image.open(input_file) as image:
        pil_image = image.convert("RGB")
        width, height = pil_image.size

    batch = [BatchInputItem(image=pil_image, prompt_type=prompt_type)]
    infer_kwargs: dict[str, Any] = {
        "include_images": include_images,
        "include_headers_footers": include_headers_footers,
    }
    if method == "vllm" and vllm_api_base:
        infer_kwargs["vllm_api_base"] = vllm_api_base

    results = runtime.manager.generate(batch, **infer_kwargs)
    if not results:
        return [], "", width, height

    page_result = results[0]
    if bool(getattr(page_result, "error", False)):
        raise RuntimeError("Chandra returned an error for this page.")

    lines_payload: list[dict[str, Any]] = []
    plain_lines: list[str] = []
    for chunk in getattr(page_result, "chunks", []) or []:
        payload = _chunk_to_payload(chunk=chunk, width=width, height=height)
        if payload is None:
            continue
        lines_payload.append(payload)
        plain_lines.append(payload["text"])

    if not plain_lines:
        fallback = _normalize_text(str(getattr(page_result, "markdown", "") or ""))
        if fallback:
            lines_payload = [
                {
                    "text": fallback,
                    "bbox": [0, 0, width, height],
                    "confidence": 0.0,
                    "words": [],
                }
            ]
            plain_lines = [fallback]

    return lines_payload, "\n".join(plain_lines), width, height


class ChandraOcrEngine(OcrEngine):
    @staticmethod
    def version() -> str:
        try:
            return importlib.metadata.version("chandra-ocr")
        except Exception:
            return "unknown"

    @staticmethod
    def creator_tag(options) -> str:
        return f"Chandra {ChandraOcrEngine.version()}"

    def __str__(self) -> str:
        return f"Chandra {self.version()}"

    @staticmethod
    def languages(options) -> set[str]:
        selected = {
            str(lang).strip()
            for lang in (getattr(options, "languages", None) or [])
            if str(lang).strip()
        }
        if selected:
            return selected
        return {"eng", "rus"}

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options) -> None:
        lines_payload, plain_text, page_width, page_height = _predict_chandra_lines(input_file, options)
        language = "en"
        selected = getattr(options, "languages", None) or []
        if selected:
            language = str(selected[0]).strip() or "en"

        hocr = build_hocr(
            page_width=page_width,
            page_height=page_height,
            lines=lines_payload,
            language=language,
        )
        output_hocr.write_text(hocr, encoding="utf-8")
        output_text.write_text(plain_text, encoding="utf-8")

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options) -> None:
        tmp_hocr = output_pdf.with_suffix(".chandra.hocr")
        ChandraOcrEngine.generate_hocr(input_file, tmp_hocr, output_text, options)

        from ocrmypdf.hocrtransform import HocrTransform

        with Image.open(input_file) as image:
            dpi = image.info.get("dpi", (300, 300))[0]

        transform = HocrTransform(hocr_filename=tmp_hocr, dpi=dpi)
        transform.to_pdf(
            out_filename=output_pdf,
            image_filename=input_file,
            invisible_text=True,
        )

        if not bool(getattr(options, "chandra_debug_keep_hocr", False)):
            try:
                tmp_hocr.unlink(missing_ok=True)
            except Exception:
                log.debug("Could not delete temporary hOCR file: %s", tmp_hocr)


@hookimpl
def add_options(parser):
    group = parser.add_argument_group("Chandra", "Options for Chandra OCR engine")
    group.add_argument(
        "--chandra-method",
        default="hf",
        choices=["hf", "vllm"],
        help="Chandra backend method (default: hf).",
    )
    group.add_argument(
        "--chandra-prompt-type",
        default="ocr_layout",
        help="Chandra prompt type (default: ocr_layout).",
    )
    group.add_argument(
        "--chandra-include-images",
        action="store_true",
        help="Keep image/figure chunks in Chandra output.",
    )
    group.add_argument(
        "--chandra-include-headers-footers",
        action="store_true",
        help="Keep header/footer chunks in Chandra output.",
    )
    group.add_argument(
        "--chandra-vllm-api-base",
        default="",
        help="Optional vLLM API base URL for --chandra-method=vllm.",
    )
    group.add_argument(
        "--chandra-debug-keep-hocr",
        action="store_true",
        help="Keep intermediate hOCR file next to output PDF.",
    )


@hookimpl
def check_options(options):
    if importlib.util.find_spec("chandra") is None:
        raise MissingDependencyError(
            "chandra-ocr is not installed. Install with: pip install 'chandra-ocr[hf]'"
        )

    if importlib.util.find_spec("chandra.model") is None:
        raise MissingDependencyError(
            "chandra.model is missing. Check your chandra-ocr installation."
        )

    method = _method_from_options(options)
    if method not in {"hf", "vllm"}:
        raise BadArgsError("Unsupported --chandra-method. Use 'hf' or 'vllm'.")

    prompt_type = str(getattr(options, "chandra_prompt_type", "") or "").strip()
    if not prompt_type:
        raise BadArgsError("Empty --chandra-prompt-type is not allowed.")

    if method == "hf" and importlib.util.find_spec("torch") is None:
        raise MissingDependencyError(
            "PyTorch is required for --chandra-method=hf. Install with: pip install torch"
        )


@hookimpl
def get_ocr_engine(options=None):
    return ChandraOcrEngine()
