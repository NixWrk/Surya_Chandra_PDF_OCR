import numpy as np

from uniscan.core.pipeline import PipelineOptions, process_loaded_items, split_spread


def _img() -> np.ndarray:
    out = np.zeros((20, 40, 3), dtype=np.uint8)
    out[:, :20] = (10, 20, 30)
    out[:, 20:] = (40, 50, 60)
    return out


def test_split_spread_returns_two_pages() -> None:
    image = _img()
    pages = split_spread(image)

    assert len(pages) == 2
    assert pages[0].shape == (20, 20, 3)
    assert pages[1].shape == (20, 20, 3)


def test_process_loaded_items_without_detector() -> None:
    loaded = [("sample.png", _img())]
    options = PipelineOptions(
        detect_document=False,
        two_page_mode=True,
        postprocess_name="None",
    )
    pages = process_loaded_items(loaded, options=options)
    assert len(pages) == 2
