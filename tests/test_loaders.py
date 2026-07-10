from __future__ import annotations

import fitz
import numpy as np

from uniscan.io.loaders import (
    imwrite_unicode,
    iter_render_pdf_page_indices,
    list_supported_in_folder,
    load_input_items,
    render_pdf_page_indices,
)


def _img(value: int) -> np.ndarray:
    return np.full((24, 32, 3), value, dtype=np.uint8)


def _make_pdf(path) -> None:
    doc = fitz.open()
    try:
        for idx in range(2):
            page = doc.new_page(width=100, height=100)
            page.insert_text((10, 40), f"Page {idx + 1}")
        doc.save(str(path))
    finally:
        doc.close()


def test_list_supported_in_folder_uses_natural_sort(tmp_path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    for name, value in [("page10.png", 10), ("page2.png", 20), ("page1.png", 30)]:
        ok = imwrite_unicode(folder / name, _img(value))
        assert ok

    paths = list_supported_in_folder(folder)
    assert [path.name for path in paths] == ["page1.png", "page2.png", "page10.png"]


def test_load_input_items_preserves_input_order_and_pdf_page_names(tmp_path) -> None:
    image_path = tmp_path / "image_a.png"
    pdf_path = tmp_path / "doc_b.pdf"
    ok = imwrite_unicode(image_path, _img(80))
    assert ok
    _make_pdf(pdf_path)

    items = load_input_items([pdf_path, image_path], pdf_dpi=72)

    assert [name for name, _image in items] == [
        "doc_b.pdf [p0001]",
        "doc_b.pdf [p0002]",
        "image_a.png",
    ]


def test_iter_render_pdf_page_indices_matches_compatibility_wrapper(tmp_path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)

    streamed = list(iter_render_pdf_page_indices(pdf_path, [1, 0], dpi=72))
    materialized = render_pdf_page_indices(pdf_path, [1, 0], dpi=72)

    assert [name for name, _image in streamed] == [
        "doc.pdf [p0002]",
        "doc.pdf [p0001]",
    ]
    assert [name for name, _image in materialized] == [name for name, _image in streamed]
    for (_name_a, image_a), (_name_b, image_b) in zip(streamed, materialized, strict=True):
        assert np.array_equal(image_a, image_b)
