import numpy as np

from uniscan.session import CaptureSession


def _img(value: int = 0) -> np.ndarray:
    return np.full((10, 12, 3), value, dtype=np.uint8)


def test_session_add_move_select_remove() -> None:
    session = CaptureSession()
    a = session.add_image(name="a", image=_img(10))
    b = session.add_image(name="b", image=_img(20))
    c = session.add_image(name="c", image=_img(30))

    assert len(session) == 3
    assert [x.name for x in session.entries] == ["a", "b", "c"]

    moved = session.move(c.entry_id, -1)
    assert moved
    assert [x.name for x in session.entries] == ["a", "c", "b"]

    session.select_all(True)
    removed = session.remove_selected()
    assert removed == 3
    assert len(session) == 0


def test_session_apply_postprocess_uses_original() -> None:
    session = CaptureSession()
    entry = session.add_image(name="gray", image=_img(127))
    session.apply_postprocess("Grayscale")

    assert entry.current_image.ndim == 2
    assert entry.original_image.ndim == 3
