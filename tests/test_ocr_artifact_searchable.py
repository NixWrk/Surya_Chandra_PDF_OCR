from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uniscan.cli import main
from uniscan.export import export_pages_as_pdf
from uniscan.ocr.artifact_searchable import (
    _PlacementCandidate,
    _align_token_indices,
    _assign_lines_to_boxes,
    _bbox_reading_order_indices,
    _blend_placements_vertical,
    _build_searchable_pdf_from_text,
    _build_geometry_candidates,
    _choose_auto_candidate,
    _clean_overlay_line,
    _coalesce_text_layer_placements,
    _estimate_page_split_weights,
    _expand_lines_to_target_count,
    _geometry_boxes_in_reading_order,
    _geometry_lines_in_reading_order,
    _has_explicit_page_markers,
    _normalize_hybrid_policy,
    _normalize_alignment_token,
    _placements_from_chandra_text_aligned_to_geometry,
    _placements_from_geometry_text_with_linefit,
    _placements_from_surya_geometry,
    _parse_artifact_filename,
    _should_blend_primary_candidate,
    _should_center_overlay_line,
    _sort_text_layer_placements,
    _split_line_to_token_boxes,
    _split_line_to_word_fragments,
    _split_page_text_lines,
    _split_lines_to_pages_by_weights,
    _split_text_to_pages_by_token_weights,
    _split_text_to_pages,
    build_compare_txt_from_benchmark,
    run_artifact_searchable_package,
)


def _build_sample_pdf(tmp_path: Path, name: str, page_values: list[int]) -> Path:
    pages: list[np.ndarray] = []
    for value in page_values:
        pages.append(np.full((200, 300, 3), value, dtype=np.uint8))
    pdf_path = tmp_path / f"{name}.pdf"
    export_pages_as_pdf(pages, out_pdf=pdf_path, dpi=120)
    return pdf_path


def _rotate_pdf_90(source_pdf: Path, out_pdf: Path) -> Path:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    for page in reader.pages:
        page.rotate(90)
        writer.add_page(page)
    with out_pdf.open("wb") as fh:
        writer.write(fh)
    return out_pdf


def _extract_pdf_text(pdf_path: Path) -> str:
    import fitz  # type: ignore

    doc = fitz.open(str(pdf_path))
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _build_text_pdf(tmp_path: Path, name: str, page_texts: list[str]) -> Path:
    import fitz  # type: ignore

    pdf_path = tmp_path / f"{name}.pdf"
    doc = fitz.open()
    try:
        for text in page_texts:
            page = doc.new_page(width=300, height=200)
            page.insert_text((20, 100), text, fontsize=12)
        doc.save(str(pdf_path))
    finally:
        doc.close()
    return pdf_path


def test_parse_artifact_filename() -> None:
    document, engine = _parse_artifact_filename(Path("ГОСТ__chandra.txt"))
    assert document == "ГОСТ"
    assert engine == "chandra"

    with pytest.raises(ValueError):
        _parse_artifact_filename(Path("broken_name.txt"))


def test_split_text_to_pages_with_markers() -> None:
    text = "\n".join(
        [
            "[SOURCE PAGE 1]",
            "Page one text",
            "[SOURCE PAGE 2]",
            "Page two text",
        ]
    )
    pages = _split_text_to_pages(text, 2)
    assert pages == ["Page one text", "Page two text"]


def test_split_lines_to_pages_by_weights() -> None:
    lines = [f"L{i}" for i in range(12)]
    pages = _split_lines_to_pages_by_weights(lines, page_count=3, page_weights=[1.0, 5.0, 1.0])
    counts = [len(page.splitlines()) if page else 0 for page in pages]
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]
    assert sum(counts) == 12


def test_split_text_to_pages_by_token_weights_prefers_heavier_pages() -> None:
    text = " ".join(f"T{i}" for i in range(24))
    pages = _split_text_to_pages_by_token_weights(
        text,
        page_count=3,
        page_weights=[1.0, 4.0, 1.0],
        line_token_span=6,
    )
    token_counts = [len(page.split()) for page in pages]
    assert token_counts[1] > token_counts[0]
    assert token_counts[1] > token_counts[2]
    assert sum(token_counts) == 24


def test_clean_overlay_line_removes_ocr_hyphen_artifacts_without_breaking_compounds() -> None:
    assert _clean_overlay_line("spatial descripti\u2014on") == "spatial description"
    assert _clean_overlay_line("the environm-ent") == "the environment"
    assert _clean_overlay_line("late-blind room-size") == "late-blind room-size"


def test_split_page_text_lines_dehyphenates_line_breaks() -> None:
    lines = _split_page_text_lines("physical move-\nment and spatial descripti\u2014on")
    assert lines == ["physical movement and spatial description"]


def test_estimate_page_split_weights_clips_outliers() -> None:
    weights = _estimate_page_split_weights(
        [
            [(0.0, 0.0, 1.0, 1.0)] * 10,
            [(0.0, 0.0, 1.0, 1.0)] * 200,
            [],
        ]
    )
    assert len(weights) == 3
    assert weights[1] < 200.0
    assert weights[2] > 0.0


def test_assign_lines_to_boxes_balances_lines() -> None:
    lines = ["L1", "L2", "L3", "L4", "L5"]
    boxes = [(0.0, 0.0, 100.0, 20.0), (0.0, 25.0, 100.0, 45.0)]
    placements = _assign_lines_to_boxes(lines, boxes)
    assert len(placements) == 2
    assert "L1" in placements[0][1]
    assert "L5" in placements[1][1]


def test_split_line_to_word_fragments_splits_into_tokens() -> None:
    parts = _split_line_to_word_fragments(
        "ИНСТРУМЕНТЫ МЕДИЦИНСКИЕ МЕТАЛЛИЧЕСКИЕ",
        bbox=(10.0, 20.0, 210.0, 40.0),
    )
    assert len(parts) == 5
    assert parts[0][1].startswith("ИНСТРУМЕНТЫ")
    assert parts[1][1] == " "
    assert parts[-1][1].strip() == "МЕТАЛЛИЧЕСКИЕ"
    assert parts[0][0][0] < parts[1][0][0] < parts[2][0][0]


def test_split_line_to_word_fragments_keeps_single_token() -> None:
    parts = _split_line_to_word_fragments("ГОСТ19126", bbox=(0.0, 0.0, 100.0, 20.0))
    assert len(parts) == 1
    assert parts[0][1] == "ГОСТ19126"


def test_split_line_to_token_boxes_keeps_token_count() -> None:
    parts = _split_line_to_token_boxes(
        "ИНСТРУМЕНТЫ МЕДИЦИНСКИЕ МЕТАЛЛИЧЕСКИЕ",
        bbox=(10.0, 20.0, 210.0, 40.0),
    )
    assert len(parts) == 3
    assert parts[0][1] == "ИНСТРУМЕНТЫ"
    assert parts[-1][1] == "МЕТАЛЛИЧЕСКИЕ"
    assert parts[0][0][0] < parts[1][0][0] < parts[2][0][0]


def test_normalize_alignment_token_folds_latin_cyrillic_ocr_noise() -> None:
    assert _normalize_alignment_token("FOCT") == _normalize_alignment_token("ГОСТ")
    assert _normalize_alignment_token("СЭВ") == _normalize_alignment_token("CEB")


def test_align_token_indices_matches_monotonic_sequence() -> None:
    src = ["gost", "19126", "79", "medicinskie"]
    dst = ["gost", "19126", "79", "medicinskie"]
    aligned, coverage, score = _align_token_indices(source_tokens=src, target_tokens=dst)
    assert coverage == pytest.approx(1.0)
    assert aligned == [0, 1, 2, 3]
    assert score > 0


def test_normalize_hybrid_policy_accepts_aliases() -> None:
    assert _normalize_hybrid_policy(None) == "auto"
    assert _normalize_hybrid_policy("surya") == "surya_only"
    assert _normalize_hybrid_policy("soft-line") == "softline"
    with pytest.raises(ValueError):
        _normalize_hybrid_policy("unknown")


def test_placements_from_chandra_text_aligned_to_geometry_uses_geometry_boxes() -> None:
    page_lines = [
        "ИНСТРУМЕНТЫ МЕДИЦИНСКИЕ",
        "МЕТАЛЛИЧЕСКИЕ",
    ]
    page_data = {
        "image_width": 1000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "ИНСТРУМЕНТЫ", "bbox": [100.0, 100.0, 300.0, 140.0]},
            {"text": "МЕДИЦИНСКИЕ", "bbox": [310.0, 100.0, 600.0, 140.0]},
            {"text": "МЕТАЛЛИЧЕСКИЕ", "bbox": [120.0, 180.0, 560.0, 220.0]},
        ],
    }
    placements, coverage = _placements_from_chandra_text_aligned_to_geometry(
        page_lines=page_lines,
        page_data=page_data,
        page_width=1000.0,
        page_height=1000.0,
    )
    assert coverage > 0.8
    assert len(placements) >= 3
    # first token should be near first geometry box
    first_bbox, first_text = placements[0]
    assert "ИНСТРУМЕНТЫ" in first_text
    assert first_bbox[0] == pytest.approx(100.0, abs=2.0)


def test_placements_from_chandra_text_aligned_to_geometry_handles_spread_rowwise_order() -> None:
    page_lines = ["L1 R1", "L2 R2"]
    page_data = {
        "image_width": 2000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "L1", "bbox": [100.0, 100.0, 300.0, 140.0]},
            {"text": "R1", "bbox": [1200.0, 100.0, 1400.0, 140.0]},
            {"text": "L2", "bbox": [100.0, 220.0, 300.0, 260.0]},
            {"text": "R2", "bbox": [1200.0, 220.0, 1400.0, 260.0]},
        ],
    }
    placements, coverage = _placements_from_chandra_text_aligned_to_geometry(
        page_lines=page_lines,
        page_data=page_data,
        page_width=1000.0,
        page_height=500.0,
    )
    assert coverage > 0.95
    texts = [text.strip() for _, text in placements if text.strip()]
    assert texts[:4] == ["L1", "R1", "L2", "R2"]


def test_build_geometry_candidates_prefers_secondary_when_primary_weak() -> None:
    page_lines = ["INSTRUMENTS MEDICAL", "METALLIC"]
    primary_page = {
        "image_width": 1000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "foo bar", "bbox": [100.0, 100.0, 380.0, 145.0]},
            {"text": "baz", "bbox": [100.0, 190.0, 300.0, 235.0]},
        ],
    }
    secondary_page = {
        "image_width": 1000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "INSTRUMENTS", "bbox": [100.0, 100.0, 320.0, 145.0]},
            {"text": "MEDICAL", "bbox": [340.0, 100.0, 600.0, 145.0]},
            {"text": "METALLIC", "bbox": [100.0, 190.0, 500.0, 235.0]},
        ],
    }
    line_boxes = [(95.0, 95.0, 610.0, 150.0), (95.0, 186.0, 510.0, 240.0)]
    candidates = _build_geometry_candidates(
        page_lines=page_lines,
        page_width=1000.0,
        page_height=1000.0,
        line_boxes=line_boxes,
        primary_page_data=primary_page,
        secondary_page_data=secondary_page,
    )
    assert candidates
    assert candidates[0].source == "secondary"
    assert candidates[0].coverage >= 0.8


def test_blend_placements_vertical_moves_tokens_towards_reference_lines() -> None:
    placements = [
        ((10.0, 100.0, 60.0, 120.0), "A"),
        ((10.0, 200.0, 70.0, 220.0), "B"),
    ]
    reference_boxes = [
        (0.0, 110.0, 150.0, 130.0),
        (0.0, 210.0, 150.0, 230.0),
    ]
    blended = _blend_placements_vertical(
        placements=placements,
        reference_boxes=reference_boxes,
        page_height=1000.0,
    )
    assert len(blended) == 2
    assert blended[0][0][0] == pytest.approx(10.0)
    assert blended[0][0][2] == pytest.approx(60.0)
    assert blended[0][0][1] > 100.0
    assert blended[0][0][3] > 120.0


def test_choose_auto_candidate_prefers_primary_on_near_tie() -> None:
    secondary = _PlacementCandidate(
        source="secondary",
        strategy="align",
        placements=[((0.0, 0.0, 10.0, 10.0), "A")],
        coverage=1.0,
        line_fit=0.0,
        token_ratio=1.0,
        score=0.77,
    )
    primary = _PlacementCandidate(
        source="primary",
        strategy="align",
        placements=[((0.0, 0.0, 10.0, 10.0), "A")],
        coverage=0.93,
        line_fit=0.0,
        token_ratio=1.0,
        score=0.756,
    )

    chosen, overridden = _choose_auto_candidate([secondary, primary])
    assert chosen is primary
    assert overridden is True


def test_should_blend_primary_candidate_only_for_weak_coverage() -> None:
    secondary = _PlacementCandidate(
        source="secondary",
        strategy="align",
        placements=[((0.0, 0.0, 10.0, 10.0), "A")],
        coverage=1.0,
        line_fit=1.0,
        token_ratio=1.0,
        score=0.95,
    )
    weak_primary = _PlacementCandidate(
        source="primary",
        strategy="assign",
        placements=[((0.0, 0.0, 10.0, 10.0), "A")],
        coverage=0.35,
        line_fit=0.0,
        token_ratio=1.0,
        score=0.15,
    )
    strong_primary = _PlacementCandidate(
        source="primary",
        strategy="align",
        placements=[((0.0, 0.0, 10.0, 10.0), "A")],
        coverage=0.82,
        line_fit=0.85,
        token_ratio=1.0,
        score=0.86,
    )

    assert _should_blend_primary_candidate(chosen=weak_primary, secondary_best=secondary) is True
    assert _should_blend_primary_candidate(chosen=strong_primary, secondary_best=secondary) is False


def test_expand_lines_to_target_count_splits_long_lines() -> None:
    source = ["A B C D E F G H", "I J K L"]
    expanded = _expand_lines_to_target_count(source, target_count=5)
    assert len(expanded) == 5
    assert "A B" in " ".join(expanded)
    assert "K L" in " ".join(expanded)


def test_assign_lines_to_boxes_merges_row_segments() -> None:
    lines = ["L1", "L2"]
    boxes = [
        (0.0, 0.0, 20.0, 10.0),
        (24.0, 1.0, 42.0, 11.0),
        (0.0, 20.0, 25.0, 30.0),
        (30.0, 21.0, 45.0, 31.0),
    ]
    placements = _assign_lines_to_boxes(lines, boxes)
    assert len(placements) == 2
    assert placements[0][0][2] >= 40.0
    assert placements[1][0][2] >= 40.0


def test_assign_lines_to_boxes_spreads_assignments_when_many_boxes() -> None:
    lines = ["L1", "L2", "L3"]
    boxes = [
        (0.0, 0.0, 30.0, 10.0),
        (0.0, 15.0, 30.0, 25.0),
        (0.0, 30.0, 30.0, 40.0),
        (0.0, 45.0, 30.0, 55.0),
        (0.0, 60.0, 30.0, 70.0),
        (0.0, 75.0, 30.0, 85.0),
    ]
    placements = _assign_lines_to_boxes(lines, boxes)
    assert len(placements) == 3
    y_positions = [item[0][1] for item in placements]
    assert y_positions[0] <= 1.0
    assert y_positions[-1] >= 70.0


def test_assign_lines_to_boxes_keeps_portrait_columns_in_reading_order() -> None:
    lines = ["L1", "L2", "L3", "R1", "R2", "R3"]
    boxes = [
        (20.0, 10.0, 180.0, 24.0),
        (320.0, 10.0, 480.0, 24.0),
        (20.0, 32.0, 180.0, 46.0),
        (320.0, 32.0, 480.0, 46.0),
        (20.0, 54.0, 180.0, 68.0),
        (320.0, 54.0, 480.0, 68.0),
    ]

    placements = _assign_lines_to_boxes(lines, boxes)

    assert [text for _bbox, text in placements] == lines
    assert [bbox[0] for bbox, _text in placements] == [20.0, 20.0, 20.0, 320.0, 320.0, 320.0]


def test_bbox_reading_order_handles_tight_gutter_with_header() -> None:
    labels = ["page", "header", "L1", "R1", "L2", "R2", "L3", "R3"]
    boxes = [
        (83.0, 74.0, 118.0, 90.0),
        (891.0, 74.0, 1210.0, 91.0),
        (85.0, 112.0, 629.0, 132.0),
        (666.0, 115.0, 1209.0, 132.0),
        (86.0, 136.0, 629.0, 153.0),
        (666.0, 136.0, 1107.0, 153.0),
        (85.0, 157.0, 629.0, 175.0),
        (686.0, 159.0, 1209.0, 176.0),
    ]

    order = _bbox_reading_order_indices(boxes, page_width=1292.0)

    assert [labels[idx] for idx in order] == ["page", "L1", "L2", "L3", "header", "R1", "R2", "R3"]


def test_bbox_reading_order_keeps_headers_inside_spread_pages() -> None:
    labels = ["R-header", "L-header", "L1", "R1", "L2", "R2"]
    boxes = [
        (700.0, 20.0, 820.0, 38.0),
        (80.0, 28.0, 200.0, 46.0),
        (82.0, 80.0, 420.0, 100.0),
        (702.0, 80.0, 1040.0, 100.0),
        (82.0, 120.0, 420.0, 140.0),
        (702.0, 120.0, 1040.0, 140.0),
    ]

    order = _bbox_reading_order_indices(boxes, page_width=1120.0)

    assert [labels[idx] for idx in order] == ["L-header", "L1", "L2", "R-header", "R1", "R2"]


def test_bbox_reading_order_uses_center_gutter_before_vertical_position() -> None:
    labels = [
        "R-title-1",
        "R-title-2",
        "R-title-3",
        "L-title-1",
        "L-title-2",
        "L-title-3",
        "L-title-4",
        "L-body-1",
        "L-body-2",
        "R-body-1",
        "R-body-2",
    ]
    boxes = [
        (421.2, 24.8, 494.3, 34.1),
        (497.2, 24.8, 528.7, 34.1),
        (531.6, 24.8, 549.1, 34.1),
        (36.1, 30.2, 52.8, 39.1),
        (55.6, 30.2, 128.9, 39.1),
        (131.7, 30.2, 159.2, 39.1),
        (161.9, 30.2, 192.2, 39.1),
        (36.1, 48.7, 98.5, 55.9),
        (101.0, 48.7, 130.2, 55.9),
        (309.3, 46.1, 345.7, 52.9),
        (348.3, 46.1, 353.5, 52.9),
    ]

    order = _bbox_reading_order_indices(boxes, page_width=589.0)

    ordered_labels = [labels[idx] for idx in order]
    assert ordered_labels.index("L-title-1") < ordered_labels.index("R-title-1")
    assert ordered_labels.index("L-body-2") < ordered_labels.index("R-title-1")


def test_bbox_reading_order_handles_near_center_spread_gutter() -> None:
    labels = ["R-header", "L-header", "R1", "L1", "R2", "L2", "L3", "L4"]
    boxes = [
        (485.7, 18.2, 559.4, 31.8),
        (65.2, 24.8, 137.6, 37.3),
        (322.7, 39.8, 560.0, 69.8),
        (63.5, 44.5, 299.2, 77.0),
        (322.7, 69.1, 560.0, 99.1),
        (63.5, 76.4, 299.2, 109.3),
        (63.5, 108.6, 299.2, 141.5),
        (63.5, 140.8, 299.2, 173.4),
    ]

    order = _bbox_reading_order_indices(boxes, page_width=589.0)

    assert [labels[idx] for idx in order] == ["L-header", "L1", "L2", "L3", "L4", "R-header", "R1", "R2"]


def test_bbox_reading_order_handles_four_columns_on_one_spread() -> None:
    labels = ["C1A", "C2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B"]
    boxes = [
        (60.0, 100.0, 170.0, 118.0),
        (210.0, 100.0, 320.0, 118.0),
        (560.0, 100.0, 670.0, 118.0),
        (710.0, 100.0, 820.0, 118.0),
        (60.0, 140.0, 170.0, 158.0),
        (210.0, 140.0, 320.0, 158.0),
        (560.0, 140.0, 670.0, 158.0),
        (710.0, 140.0, 820.0, 158.0),
    ]

    order = _bbox_reading_order_indices(boxes, page_width=900.0)

    assert [labels[idx] for idx in order] == ["C1A", "C1B", "C2A", "C2B", "C3A", "C3B", "C4A", "C4B"]


def test_should_center_overlay_line_only_for_centered_heading_like_text() -> None:
    centered_bbox = (100.0, 100.0, 500.0, 130.0)

    assert _should_center_overlay_line(
        "СОЮЗА ССР",
        raw_horiz_scale=260.0,
        bbox=centered_bbox,
        page_width=600.0,
    )
    assert _should_center_overlay_line(
        "Москва",
        raw_horiz_scale=360.0,
        bbox=centered_bbox,
        page_width=600.0,
    )
    assert not _should_center_overlay_line(
        "called it his",
        raw_horiz_scale=360.0,
        bbox=(40.0, 100.0, 300.0, 130.0),
        page_width=600.0,
    )
    assert not _should_center_overlay_line(
        "The patient who has a high-school education is a very poor reader",
        raw_horiz_scale=110.0,
        bbox=centered_bbox,
        page_width=600.0,
    )


def test_placements_from_surya_geometry_scales_and_cleans_text() -> None:
    payload = {
        "image_width": 1000.0,
        "image_height": 2000.0,
        "lines": [
            {
                "text": "<b>ИНСТРУМЕНТЫ</b>",
                "bbox": [100.0, 200.0, 300.0, 260.0],
            }
        ],
    }
    placements = _placements_from_surya_geometry(
        page_data=payload,
        page_width=500.0,
        page_height=1000.0,
    )
    assert len(placements) == 1
    bbox, text = placements[0]
    assert text == "ИНСТРУМЕНТЫ"
    assert bbox == pytest.approx((50.0, 100.0, 150.0, 130.0))


def test_placements_from_surya_geometry_auto_spread_orders_left_then_right() -> None:
    payload = {
        "image_width": 2000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "L-top", "bbox": [80.0, 80.0, 900.0, 120.0]},
            {"text": "R-top", "bbox": [1100.0, 85.0, 1900.0, 125.0]},
            {"text": "L-bottom", "bbox": [80.0, 820.0, 900.0, 860.0]},
            {"text": "R-bottom", "bbox": [1100.0, 825.0, 1900.0, 865.0]},
            {"text": "L-mid", "bbox": [80.0, 450.0, 900.0, 490.0]},
            {"text": "R-mid", "bbox": [1100.0, 455.0, 1900.0, 495.0]},
        ],
    }
    placements = _placements_from_surya_geometry(
        page_data=payload,
        page_width=1000.0,
        page_height=500.0,
    )
    ordered_texts = [text for _, text in placements]
    assert ordered_texts == ["L-top", "L-mid", "L-bottom", "R-top", "R-mid", "R-bottom"]


def test_placements_from_surya_geometry_orders_portrait_columns_left_then_right() -> None:
    payload = {
        "image_width": 1000.0,
        "image_height": 1600.0,
        "lines": [
            {"text": "L1", "bbox": [80.0, 100.0, 420.0, 140.0]},
            {"text": "R1", "bbox": [580.0, 100.0, 920.0, 140.0]},
            {"text": "L2", "bbox": [80.0, 240.0, 420.0, 280.0]},
            {"text": "R2", "bbox": [580.0, 240.0, 920.0, 280.0]},
            {"text": "L3", "bbox": [80.0, 380.0, 420.0, 420.0]},
            {"text": "R3", "bbox": [580.0, 380.0, 920.0, 420.0]},
        ],
    }

    placements = _placements_from_surya_geometry(
        page_data=payload,
        page_width=500.0,
        page_height=800.0,
    )

    assert [text for _bbox, text in placements] == ["L1", "L2", "L3", "R1", "R2", "R3"]


def test_geometry_lines_in_reading_order_auto_spread() -> None:
    payload = {
        "image_width": 2000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "L-top", "bbox": [80.0, 80.0, 900.0, 120.0]},
            {"text": "R-top", "bbox": [1100.0, 85.0, 1900.0, 125.0]},
            {"text": "L-bottom", "bbox": [80.0, 820.0, 900.0, 860.0]},
            {"text": "R-bottom", "bbox": [1100.0, 825.0, 1900.0, 865.0]},
            {"text": "L-mid", "bbox": [80.0, 450.0, 900.0, 490.0]},
            {"text": "R-mid", "bbox": [1100.0, 455.0, 1900.0, 495.0]},
        ],
    }
    lines = _geometry_lines_in_reading_order(
        page_data=payload,
        page_width=1000.0,
        page_height=500.0,
    )
    assert lines == ["L-top", "L-mid", "L-bottom", "R-top", "R-mid", "R-bottom"]


def test_geometry_boxes_in_reading_order_auto_spread() -> None:
    payload = {
        "image_width": 2000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "L-top", "bbox": [80.0, 80.0, 900.0, 120.0]},
            {"text": "R-top", "bbox": [1100.0, 85.0, 1900.0, 125.0]},
            {"text": "L-bottom", "bbox": [80.0, 820.0, 900.0, 860.0]},
            {"text": "R-bottom", "bbox": [1100.0, 825.0, 1900.0, 865.0]},
            {"text": "L-mid", "bbox": [80.0, 450.0, 900.0, 490.0]},
            {"text": "R-mid", "bbox": [1100.0, 455.0, 1900.0, 495.0]},
        ],
    }
    boxes = _geometry_boxes_in_reading_order(
        page_data=payload,
        page_width=1000.0,
        page_height=500.0,
    )
    assert len(boxes) == 6
    y_positions = [item[1] for item in boxes]
    assert y_positions[:3] == sorted(y_positions[:3])
    assert y_positions[3:] == sorted(y_positions[3:])


def test_placements_from_geometry_text_with_linefit_prefers_detected_boxes() -> None:
    payload = {
        "image_width": 1000.0,
        "image_height": 1000.0,
        "lines": [
            {"text": "LINE A", "bbox": [100.0, 100.0, 900.0, 160.0]},
            {"text": "LINE B", "bbox": [100.0, 200.0, 900.0, 260.0]},
        ],
    }
    line_boxes = [
        (10.0, 20.0, 120.0, 36.0),
        (12.0, 40.0, 125.0, 56.0),
    ]
    placements = _placements_from_geometry_text_with_linefit(
        page_data=payload,
        page_width=500.0,
        page_height=500.0,
        line_boxes=line_boxes,
    )
    assert len(placements) == 2
    assert placements[0][0] == pytest.approx(line_boxes[0])
    assert placements[1][0] == pytest.approx(line_boxes[1])
    assert [text for _, text in placements] == ["LINE A", "LINE B"]


def test_placements_from_geometry_text_with_linefit_falls_back_to_geometry() -> None:
    payload = {
        "image_width": 1000.0,
        "image_height": 2000.0,
        "lines": [{"text": "SINGLE", "bbox": [100.0, 200.0, 300.0, 260.0]}],
    }
    placements = _placements_from_geometry_text_with_linefit(
        page_data=payload,
        page_width=500.0,
        page_height=1000.0,
        line_boxes=[],
    )
    assert len(placements) == 1
    bbox, text = placements[0]
    assert text == "SINGLE"
    assert bbox == pytest.approx((50.0, 100.0, 150.0, 130.0))


def test_build_searchable_pdf_keeps_text_when_boxes_are_tiny(monkeypatch, tmp_path: Path) -> None:
    src_pdf = _build_sample_pdf(tmp_path, "tiny_box_fixture", [40])
    out_pdf = tmp_path / "tiny_box_out.pdf"
    text = "\n".join(f"Line {idx:03d}" for idx in range(120))

    monkeypatch.setattr(
        "uniscan.ocr.artifact_searchable._estimate_page_line_bboxes",
        lambda **_kwargs: [(0.0, 0.0, 40.0, 12.0)],
    )

    _build_searchable_pdf_from_text(
        source_pdf=src_pdf,
        text=text,
        out_pdf=out_pdf,
    )
    extracted = _extract_pdf_text(out_pdf)
    assert "Line 000" in extracted
    assert "Line 119" in extracted


def test_build_searchable_pdf_wraps_overcompressed_overlay_lines(monkeypatch, tmp_path: Path) -> None:
    src_pdf = _build_sample_pdf(tmp_path, "compressed_overlay_fixture", [40])
    out_pdf = tmp_path / "compressed_overlay_out.pdf"
    text = (
        "[SOURCE PAGE 1]\n"
        "Исполняют его следующим образом. На светочувствительную бумагу кладут рисунок "
        "или печатный лист, прикрывают стеклом и выставляют на дневной свет.\n"
    )

    monkeypatch.setattr(
        "uniscan.ocr.artifact_searchable._estimate_page_line_bboxes",
        lambda **_kwargs: [(20.0, 20.0, 160.0, 52.0)],
    )

    _build_searchable_pdf_from_text(
        source_pdf=src_pdf,
        text=text,
        out_pdf=out_pdf,
    )

    extracted = _normalize_ws(_extract_pdf_text(out_pdf))
    assert "Исполняют его следующим образом" in extracted
    assert "светочувствительную бумагу" in extracted

    from pypdf import PdfReader
    from pypdf.generic import ContentStream

    reader = PdfReader(str(out_pdf))
    content = ContentStream(reader.pages[0].get_contents(), reader)
    horiz_scales = [
        float(operands[0])
        for operands, operator in content.operations
        if operator == b"Tz" and operands
    ]
    assert horiz_scales
    assert min(horiz_scales) >= 64.0


def test_build_searchable_pdf_orders_hybrid_text_layer_by_geometry_columns(
    monkeypatch, tmp_path: Path
) -> None:
    src_pdf = _build_sample_pdf(tmp_path, "column_order_fixture", [40])
    out_pdf = tmp_path / "column_order_out.pdf"
    text = "\n".join(
        [
            "[SOURCE PAGE 1]",
            "LEFT-ONE",
            "RIGHT-ONE",
            "LEFT-TWO",
            "RIGHT-TWO",
        ]
    )

    monkeypatch.setattr(
        "uniscan.ocr.artifact_searchable._estimate_page_line_bboxes",
        lambda **_kwargs: [],
    )

    _build_searchable_pdf_from_text(
        source_pdf=src_pdf,
        text=text,
        out_pdf=out_pdf,
        surya_geometry_by_page={
            1: {
                "image_width": 1000.0,
                "image_height": 1000.0,
                "lines": [
                    {"text": "LEFT-ONE", "bbox": [100.0, 100.0, 380.0, 150.0]},
                    {"text": "RIGHT-ONE", "bbox": [620.0, 100.0, 900.0, 150.0]},
                    {"text": "LEFT-TWO", "bbox": [100.0, 200.0, 380.0, 250.0]},
                    {"text": "RIGHT-TWO", "bbox": [620.0, 200.0, 900.0, 250.0]},
                ],
            }
        },
        geometry_linefit_prefer=True,
    )

    extracted = _normalize_ws(_extract_pdf_text(out_pdf))
    assert extracted.index("LEFT-ONE") < extracted.index("LEFT-TWO")
    assert extracted.index("LEFT-TWO") < extracted.index("RIGHT-ONE")
    assert extracted.index("RIGHT-ONE") < extracted.index("RIGHT-TWO")


def test_build_searchable_pdf_orders_four_column_spread_text_layer(
    monkeypatch, tmp_path: Path
) -> None:
    src_pdf = _build_sample_pdf(tmp_path, "four_column_spread_fixture", [40])
    out_pdf = tmp_path / "four_column_spread_out.pdf"
    text = "\n".join(
        [
            "[SOURCE PAGE 1]",
            "C1A",
            "C2A",
            "C3A",
            "C4A",
            "C1B",
            "C2B",
            "C3B",
            "C4B",
        ]
    )

    monkeypatch.setattr(
        "uniscan.ocr.artifact_searchable._estimate_page_line_bboxes",
        lambda **_kwargs: [],
    )

    _build_searchable_pdf_from_text(
        source_pdf=src_pdf,
        text=text,
        out_pdf=out_pdf,
        surya_geometry_by_page={
            1: {
                "image_width": 900.0,
                "image_height": 900.0,
                "lines": [
                    {"text": "C1A", "bbox": [60.0, 100.0, 170.0, 118.0]},
                    {"text": "C2A", "bbox": [210.0, 100.0, 320.0, 118.0]},
                    {"text": "C3A", "bbox": [560.0, 100.0, 670.0, 118.0]},
                    {"text": "C4A", "bbox": [710.0, 100.0, 820.0, 118.0]},
                    {"text": "C1B", "bbox": [60.0, 140.0, 170.0, 158.0]},
                    {"text": "C2B", "bbox": [210.0, 140.0, 320.0, 158.0]},
                    {"text": "C3B", "bbox": [560.0, 140.0, 670.0, 158.0]},
                    {"text": "C4B", "bbox": [710.0, 140.0, 820.0, 158.0]},
                ],
            }
        },
        geometry_linefit_prefer=True,
    )

    extracted = _normalize_ws(_extract_pdf_text(out_pdf))
    expected = ["C1A", "C1B", "C2A", "C2B", "C3A", "C3B", "C4A", "C4B"]
    positions = [extracted.index(item) for item in expected]
    assert positions == sorted(positions)


def test_build_geometry_candidates_prefers_geometry_text_when_source_order_is_mixed() -> None:
    page_lines = [
        "R-header",
        "L-header",
        "R-one epsilon zeta",
        "L-one alpha beta",
        "R-two eta theta",
        "L-two gamma delta",
    ]
    page_data = {
        "image_width": 600.0,
        "image_height": 400.0,
        "lines": [
            {"text": "L-header", "bbox": [70.0, 30.0, 160.0, 45.0]},
            {"text": "L-one alpha beta", "bbox": [70.0, 70.0, 310.0, 90.0]},
            {"text": "L-two gamma delta", "bbox": [70.0, 105.0, 310.0, 125.0]},
            {"text": "R-header", "bbox": [485.0, 25.0, 560.0, 40.0]},
            {"text": "R-one epsilon zeta", "bbox": [330.0, 65.0, 560.0, 85.0]},
            {"text": "R-two eta theta", "bbox": [330.0, 100.0, 560.0, 120.0]},
        ],
    }

    candidates = _build_geometry_candidates(
        page_lines=page_lines,
        page_width=600.0,
        page_height=400.0,
        line_boxes=[],
        primary_page_data=page_data,
        secondary_page_data=None,
    )
    chosen, _override = _choose_auto_candidate(candidates)

    assert chosen is not None
    assert chosen.strategy == "linefit"
    ordered_texts = [
        text
        for _bbox, text in _coalesce_text_layer_placements(
            _sort_text_layer_placements(chosen.placements, page_width=600.0)
        )
    ]
    assert ordered_texts == [
        "L-header",
        "L-one alpha beta",
        "L-two gamma delta",
        "R-header",
        "R-one epsilon zeta",
        "R-two eta theta",
    ]


def test_build_searchable_pdf_normalizes_rotated_pages(tmp_path: Path) -> None:
    base_pdf = _build_sample_pdf(tmp_path, "rotated_fixture_base", [80])
    rotated_pdf = _rotate_pdf_90(base_pdf, tmp_path / "rotated_fixture.pdf")
    out_pdf = tmp_path / "rotated_out.pdf"

    _build_searchable_pdf_from_text(
        source_pdf=rotated_pdf,
        text="[SOURCE PAGE 1]\nROTATED PAGE TEXT\n",
        out_pdf=out_pdf,
        surya_geometry_by_page={
            1: {
                "image_width": 300.0,
                "image_height": 200.0,
                "lines": [{"text": "ROTATED PAGE TEXT", "bbox": [30.0, 40.0, 260.0, 85.0]}],
            }
        },
    )

    from pypdf import PdfReader

    reader = PdfReader(str(out_pdf))
    assert int(reader.pages[0].get("/Rotate", 0) or 0) == 0
    extracted = _extract_pdf_text(out_pdf)
    assert "ROTATED PAGE TEXT" in extracted


def test_run_artifact_searchable_package_builds_pdfs(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    _build_sample_pdf(pdf_root, "ГОСТ с плохим качеством скана", [30, 90])
    (compare_dir / "ГОСТ с плохим качеством скана__chandra.txt").write_text(
        "Alpha line for page one.\n"
        "\u041f\u0440\u0438\u043c\u0435\u0440 "
        "\u0440\u0443\u0441\u0441\u043a\u043e\u0439 "
        "\u0441\u0442\u0440\u043e\u043a\u0438.",
        encoding="utf-8",
    )
    (compare_dir / "ГОСТ с плохим качеством скана__surya.txt").write_text(
        "Surya text content.",
        encoding="utf-8",
    )
    (compare_dir / "Missing Document__olmocr.txt").write_text("text", encoding="utf-8")

    results = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra", "surya", "olmocr"),
    )

    assert len(results) == 3
    ok_rows = [row for row in results if row.status == "ok"]
    err_rows = [row for row in results if row.status == "error"]
    assert len(ok_rows) == 2
    assert len(err_rows) == 1
    assert err_rows[0].engine == "olmocr"
    assert "not found" in (err_rows[0].error or "").lower()

    for row in ok_rows:
        assert row.searchable_pdf_path is not None
        pdf_path = Path(row.searchable_pdf_path)
        assert pdf_path.exists()
        extracted = _extract_pdf_text(pdf_path)
        assert extracted.strip()
        if row.engine == "chandra":
            assert any(0x0400 <= ord(ch) <= 0x04FF for ch in extracted)

    assert (output_dir / "artifact_searchable_summary.json").exists()
    assert (output_dir / "artifact_searchable_summary.csv").exists()


def test_run_artifact_searchable_package_uses_chandra_sidecar_geometry(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    doc_name = "fixture_doc"
    _build_sample_pdf(pdf_root, doc_name, [40])
    (compare_dir / f"{doc_name}__chandra.txt").write_text("", encoding="utf-8")

    chandra_dir = compare_dir.parent / "chandra"
    chandra_dir.mkdir()
    (chandra_dir / "pages.json").write_text(
        json.dumps(
            {
                "pdf_path": str((pdf_root / f"{doc_name}.pdf")),
                "engine": "chandra",
                "pages": [
                    {
                        "source_page": 1,
                        "geometry_file": "page_0001.chandra.json",
                        "geometry_type": "chandra_text_lines",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (chandra_dir / "page_0001.chandra.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_name": "00001.png",
                        "pages": [
                            {
                                "image_bbox": [0, 0, 300, 200],
                                "text_lines": [
                                    {"text": "CHANDRA GEOMETRY LINE", "bbox": [20, 20, 280, 60]}
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra",),
    )

    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].searchable_pdf_path is not None
    extracted = _extract_pdf_text(Path(rows[0].searchable_pdf_path))
    assert "CHANDRA GEOMETRY LINE" in extracted


def test_run_artifact_searchable_package_reads_nested_chandra_pages_json(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    doc_name = "fixture_doc"
    _build_sample_pdf(pdf_root, doc_name, [40])
    (compare_dir / f"{doc_name}__chandra.txt").write_text("TXT LINE SHOULD WIN", encoding="utf-8")

    nested_dir = compare_dir.parent / "chandra" / "chandra"
    nested_dir.mkdir(parents=True)
    (nested_dir / "pages.json").write_text(
        json.dumps(
            {
                "pdf_path": str((pdf_root / f"{doc_name}.pdf")),
                "engine": "chandra",
                "pages": [
                    {
                        "source_page": 1,
                        "geometry_file": "page_0001.chandra.json",
                        "geometry_type": "chandra_text_lines",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (nested_dir / "page_0001.chandra.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_name": "00001.png",
                        "pages": [
                            {
                                "image_bbox": [0, 0, 300, 200],
                                "text_lines": [
                                    {"text": "GEOMETRY LINE", "bbox": [20, 20, 280, 60]}
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra",),
    )

    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].searchable_pdf_path is not None
    extracted = _extract_pdf_text(Path(rows[0].searchable_pdf_path))
    assert "TXT LINE SHOULD WIN" in extracted


def test_run_artifact_searchable_package_uses_chandra_geometry_on_pdf_name_mismatch(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    doc_name = "fixture_doc"
    _build_sample_pdf(pdf_root, doc_name, [50])
    (compare_dir / f"{doc_name}__chandra.txt").write_text("", encoding="utf-8")

    chandra_dir = compare_dir.parent / "chandra"
    chandra_dir.mkdir()
    (chandra_dir / "pages.json").write_text(
        json.dumps(
            {
                # Intentionally mismatched stem (simulates mojibake path in JSON).
                "pdf_path": str(pdf_root / "Ð¤Ð°Ð¹Ð».pdf"),
                "engine": "chandra",
                "pages": [
                    {
                        "source_page": 1,
                        "geometry_file": "page_0001.chandra.json",
                        "geometry_type": "chandra_text_lines",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (chandra_dir / "page_0001.chandra.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_name": "00001.png",
                        "pages": [
                            {
                                "image_bbox": [0, 0, 300, 200],
                                "text_lines": [
                                    {"text": "GEOMETRY STILL APPLIED", "bbox": [30, 30, 260, 70]}
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra",),
    )

    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].searchable_pdf_path is not None
    extracted = _extract_pdf_text(Path(rows[0].searchable_pdf_path))
    assert "GEOMETRY STILL APPLIED" in extracted


def test_run_artifact_searchable_package_writes_geometry_debug_log(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    doc_name = "fixture_doc"
    _build_sample_pdf(pdf_root, doc_name, [50, 80])
    (compare_dir / f"{doc_name}__chandra.txt").write_text(
        "[SOURCE PAGE 1]\nCHANDRA PAGE ONE\n[SOURCE PAGE 2]\nCHANDRA PAGE TWO\n",
        encoding="utf-8",
    )

    def _write_pages(root: Path, geometry_type: str, prefix: str, text1: str, text2: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "pages.json").write_text(
            json.dumps(
                {
                    "pdf_path": str(pdf_root / f"{doc_name}.pdf"),
                    "engine": prefix,
                    "pages": [
                        {"source_page": 1, "geometry_file": f"page_0001.{prefix}.json", "geometry_type": geometry_type},
                        {"source_page": 2, "geometry_file": f"page_0002.{prefix}.json", "geometry_type": geometry_type},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (root / f"page_0001.{prefix}.json").write_text(
            json.dumps(
                {
                    "images": [
                        {"image_name": "00001.png", "pages": [{"image_bbox": [0, 0, 300, 200], "text_lines": [{"text": text1, "bbox": [20, 20, 280, 60]}]}]}
                    ]
                }
            ),
            encoding="utf-8",
        )
        (root / f"page_0002.{prefix}.json").write_text(
            json.dumps(
                {
                    "images": [
                        {"image_name": "00002.png", "pages": [{"image_bbox": [0, 0, 300, 200], "text_lines": [{"text": text2, "bbox": [20, 120, 280, 160]}]}]}
                    ]
                }
            ),
            encoding="utf-8",
        )

    _write_pages(
        compare_dir.parent / "chandra",
        geometry_type="chandra_text_lines",
        prefix="chandra",
        text1="CHANDRA PAGE ONE",
        text2="CHANDRA PAGE TWO",
    )
    surya_override = tmp_path / "surya_override"
    _write_pages(
        surya_override / "surya",
        geometry_type="surya_text_lines",
        prefix="surya",
        text1="SURYA PAGE ONE",
        text2="SURYA PAGE TWO",
    )

    old_geom = os.environ.get("UNISCAN_CHANDRA_GEOMETRY_DIR")
    try:
        os.environ["UNISCAN_CHANDRA_GEOMETRY_DIR"] = str(surya_override / "surya")
        rows = run_artifact_searchable_package(
            compare_dir=compare_dir,
            pdf_root=pdf_root,
            output_dir=output_dir,
            engines=("chandra",),
            chandra_geometry_policy="auto",
            geometry_debug_log=True,
        )
    finally:
        if old_geom is None:
            os.environ.pop("UNISCAN_CHANDRA_GEOMETRY_DIR", None)
        else:
            os.environ["UNISCAN_CHANDRA_GEOMETRY_DIR"] = old_geom

    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].geometry_log_path is not None
    log_path = Path(rows[0].geometry_log_path)
    assert log_path.exists()
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["policy"] == "auto"
    assert len(payload["pages"]) == 2
    assert payload["pages"][0]["page"] == 1
    assert payload["pages"][0]["candidate_count"] >= 1


def test_run_artifact_searchable_package_require_markers(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()

    _build_sample_pdf(pdf_root, "fixture_doc", [30, 90])
    (compare_dir / "fixture_doc__chandra.txt").write_text("plain markerless text", encoding="utf-8")

    results = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=output_dir,
        engines=("chandra",),
        require_page_markers=True,
    )

    assert len(results) == 1
    assert results[0].status == "error"
    assert "no explicit page markers" in (results[0].error or "").lower()


def test_run_artifact_searchable_package_delete_original_text_layer(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    out_keep = tmp_path / "out_keep"
    out_delete = tmp_path / "out_delete"
    compare_dir.mkdir()
    pdf_root.mkdir()

    doc_name = "fixture_doc"
    _build_text_pdf(
        pdf_root,
        doc_name,
        ["BASE PAGE ONE", "BASE PAGE TWO SHOULD DISAPPEAR"],
    )
    (compare_dir / f"{doc_name}__chandra.txt").write_text(
        "[SOURCE PAGE 1]\nNEW PAGE ONE OCR\n",
        encoding="utf-8",
    )

    keep_rows = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=out_keep,
        engines=("chandra",),
        require_page_markers=True,
        delete_original_text_layer=False,
    )
    assert len(keep_rows) == 1
    assert keep_rows[0].status == "ok"
    assert keep_rows[0].searchable_pdf_path is not None
    keep_text = _normalize_ws(_extract_pdf_text(Path(keep_rows[0].searchable_pdf_path)))
    assert "NEW PAGE ONE OCR" in keep_text
    assert "BASE PAGE TWO SHOULD DISAPPEAR" in keep_text

    delete_rows = run_artifact_searchable_package(
        compare_dir=compare_dir,
        pdf_root=pdf_root,
        output_dir=out_delete,
        engines=("chandra",),
        require_page_markers=True,
        delete_original_text_layer=True,
    )
    assert len(delete_rows) == 1
    assert delete_rows[0].status == "ok"
    assert delete_rows[0].searchable_pdf_path is not None
    delete_text = _normalize_ws(_extract_pdf_text(Path(delete_rows[0].searchable_pdf_path)))
    assert "NEW PAGE ONE OCR" in delete_text
    assert "BASE PAGE TWO SHOULD DISAPPEAR" not in delete_text


def test_build_compare_txt_from_benchmark(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "bench"
    output_dir = tmp_path / "compare_txt"
    benchmark_root.mkdir()

    src_txt = benchmark_root / "fixture_doc_chandra.txt"
    src_txt.write_text("[SOURCE PAGE 0001]\nHello\n", encoding="utf-8")
    payload = [
        {
            "engine": "chandra",
            "status": "ok",
            "artifact_path": str(src_txt),
        },
        {
            "engine": "surya",
            "status": "error",
            "artifact_path": "",
            "error": "Surya cache/weights preflight failed",
        },
    ]
    (benchmark_root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = build_compare_txt_from_benchmark(
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        engines=("chandra", "surya"),
    )
    assert len(rows) == 2
    ok_rows = [row for row in rows if row.status == "ok"]
    err_rows = [row for row in rows if row.status == "error"]
    assert len(ok_rows) == 1
    assert len(err_rows) == 1
    assert err_rows[0].error == "engine status is 'error': Surya cache/weights preflight failed"
    assert (output_dir / "fixture_doc__chandra.txt").exists()
    assert (output_dir / "sources_map.txt").exists()


def test_has_explicit_page_markers_detects_marker_in_multiline_text() -> None:
    text = "preface line\n[SOURCE PAGE 0001]\nbody line"
    assert _has_explicit_page_markers(text) is True


def test_build_compare_txt_from_reports_without_summary(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "bench"
    output_dir = tmp_path / "compare_txt"
    benchmark_root.mkdir()

    surya_dir = benchmark_root / "surya"
    chandra_dir = benchmark_root / "chandra"
    surya_dir.mkdir()
    chandra_dir.mkdir()

    surya_txt = surya_dir / "fixture_doc_surya.txt"
    chandra_txt = chandra_dir / "fixture_doc_chandra.txt"
    surya_txt.write_text("[SOURCE PAGE 0001]\nS\n", encoding="utf-8")
    chandra_txt.write_text("[SOURCE PAGE 0001]\nC\n", encoding="utf-8")

    surya_report = {
        "pdf_path": "fixture_doc.pdf",
        "results": [{"engine": "surya", "status": "ok", "artifact_path": str(surya_txt)}],
    }
    chandra_report = {
        "pdf_path": "fixture_doc.pdf",
        "results": [{"engine": "chandra", "status": "ok", "artifact_path": str(chandra_txt)}],
    }
    (surya_dir / "fixture_doc_ocr_benchmark.json").write_text(json.dumps(surya_report), encoding="utf-8")
    (chandra_dir / "fixture_doc_ocr_benchmark.json").write_text(json.dumps(chandra_report), encoding="utf-8")

    rows = build_compare_txt_from_benchmark(
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        engines=("surya", "chandra"),
    )
    assert len(rows) == 2
    assert all(row.status == "ok" for row in rows)
    assert (output_dir / "fixture_doc__surya.txt").exists()
    assert (output_dir / "fixture_doc__chandra.txt").exists()
    sources_map = (output_dir / "sources_map.txt").read_text(encoding="utf-8")
    assert "discovered_reports=2" in sources_map


def test_cli_build_searchable_from_artifacts_success(monkeypatch, tmp_path: Path, capsys) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()
    output_dir.mkdir()

    def fake_run(**kwargs):
        assert kwargs["compare_dir"] == compare_dir
        assert kwargs["pdf_root"] == pdf_root
        assert kwargs["output_dir"] == output_dir
        assert kwargs["engines"] == ("chandra", "surya")
        assert kwargs["require_page_markers"] is False
        assert kwargs["delete_original_text_layer"] is False
        return [
            SimpleNamespace(
                document="ГОСТ",
                engine="chandra",
                status="ok",
                source_pdf_path="x.pdf",
                text_artifact_path="x.txt",
                searchable_pdf_path="out.pdf",
                page_count=2,
                text_chars=123,
                elapsed_seconds=1.0,
                error=None,
            )
        ]

    monkeypatch.setattr("uniscan.cli.run_artifact_searchable_package", fake_run)
    monkeypatch.setattr("uniscan.cli.summarize_artifact_searchable_package", lambda rows: f"rows={len(rows)}")

    exit_code = main(
        [
            "build-searchable-from-artifacts",
            "--compare-dir",
            str(compare_dir),
            "--pdf-root",
            str(pdf_root),
            "--output",
            str(output_dir),
            "--engines",
            "chandra",
            "surya",
        ]
    )
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "rows=1" in stdout


def test_cli_build_searchable_from_artifacts_strict_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()
    output_dir.mkdir()

    def fake_run(**_kwargs):
        assert _kwargs["require_page_markers"] is True
        assert _kwargs["delete_original_text_layer"] is False
        return [
            SimpleNamespace(
                document="ГОСТ",
                engine="chandra",
                status="ok",
                source_pdf_path="x.pdf",
                text_artifact_path="x.txt",
                searchable_pdf_path="ok.pdf",
                page_count=2,
                text_chars=123,
                elapsed_seconds=1.0,
                error=None,
            ),
            SimpleNamespace(
                document="ГОСТ",
                engine="surya",
                status="error",
                source_pdf_path="x.pdf",
                text_artifact_path="y.txt",
                searchable_pdf_path=None,
                page_count=0,
                text_chars=0,
                elapsed_seconds=1.0,
                error="broken",
            ),
        ]

    monkeypatch.setattr("uniscan.cli.run_artifact_searchable_package", fake_run)
    monkeypatch.setattr("uniscan.cli.summarize_artifact_searchable_package", lambda _rows: "summary")

    exit_code = main(
        [
            "build-searchable-from-artifacts",
            "--compare-dir",
            str(compare_dir),
            "--pdf-root",
            str(pdf_root),
            "--output",
            str(output_dir),
            "--strict",
        ]
    )
    stdout = capsys.readouterr().out
    assert exit_code == 1
    assert "summary" in stdout


def test_cli_build_searchable_from_artifacts_delete_text_layer_flag(monkeypatch, tmp_path: Path, capsys) -> None:
    compare_dir = tmp_path / "compare"
    pdf_root = tmp_path / "pdf_root"
    output_dir = tmp_path / "out"
    compare_dir.mkdir()
    pdf_root.mkdir()
    output_dir.mkdir()

    def fake_run(**kwargs):
        assert kwargs["delete_original_text_layer"] is True
        return [
            SimpleNamespace(
                document="fixture",
                engine="chandra",
                status="ok",
                source_pdf_path="x.pdf",
                text_artifact_path="x.txt",
                searchable_pdf_path="ok.pdf",
                page_count=1,
                text_chars=10,
                elapsed_seconds=0.1,
                error=None,
            )
        ]

    monkeypatch.setattr("uniscan.cli.run_artifact_searchable_package", fake_run)
    monkeypatch.setattr("uniscan.cli.summarize_artifact_searchable_package", lambda rows: f"rows={len(rows)}")

    exit_code = main(
        [
            "build-searchable-from-artifacts",
            "--compare-dir",
            str(compare_dir),
            "--pdf-root",
            str(pdf_root),
            "--output",
            str(output_dir),
            "--delete-original-text-layer",
        ]
    )
    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "rows=1" in stdout


def test_cli_prepare_compare_txt_strict_fails(tmp_path: Path, capsys) -> None:
    benchmark_root = tmp_path / "bench"
    output_dir = tmp_path / "compare_txt"
    benchmark_root.mkdir()
    src_txt = benchmark_root / "fixture_doc_chandra.txt"
    src_txt.write_text("[SOURCE PAGE 0001]\nHello\n", encoding="utf-8")
    payload = [
        {"engine": "chandra", "status": "ok", "artifact_path": str(src_txt)},
        {"engine": "surya", "status": "error", "artifact_path": ""},
    ]
    (benchmark_root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "prepare-compare-txt",
            "--benchmark-root",
            str(benchmark_root),
            "--output",
            str(output_dir),
            "--engines",
            "chandra",
            "surya",
            "--strict",
        ]
    )
    stdout = capsys.readouterr().out
    assert exit_code == 1
    assert "chandra: ok" in stdout
    assert "surya: error" in stdout
