import numpy as np

from uniscan.export import export_pages_as_files, export_pages_as_pdf


def _pages() -> list[np.ndarray]:
    a = np.zeros((30, 50, 3), dtype=np.uint8)
    b = np.full((30, 50, 3), 220, dtype=np.uint8)
    return [a, b]


def test_export_pages_as_files(tmp_path) -> None:
    out = export_pages_as_files(_pages(), output_dir=tmp_path, ext="png", base_name="p")
    assert len(out) == 2
    assert out[0].exists()
    assert out[1].exists()


def test_export_pages_as_pdf(tmp_path) -> None:
    out_pdf = tmp_path / "out.pdf"
    result = export_pages_as_pdf(_pages(), out_pdf=out_pdf, dpi=200)
    assert result.exists()
    assert result.suffix.lower() == ".pdf"
    assert result.stat().st_size > 0
