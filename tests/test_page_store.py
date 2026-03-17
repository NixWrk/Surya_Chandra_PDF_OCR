import numpy as np

from uniscan.storage import PageStore


def _img() -> np.ndarray:
    out = np.zeros((40, 60, 3), dtype=np.uint8)
    out[:, :] = (10, 20, 30)
    return out


def test_page_store_add_read_remove(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    entry_id = "entry_a"
    original_path, current_path, thumb_path = store.add_page(entry_id, _img())

    assert original_path.exists()
    assert current_path.exists()
    assert thumb_path.exists()
    assert store.read_image(current_path).shape == (40, 60, 3)

    store.remove_page(entry_id)
    assert not original_path.exists()
    assert not current_path.exists()


def test_page_store_cleanup_session(tmp_path) -> None:
    store = PageStore(root_dir=tmp_path)
    store.add_page("entry_b", _img())
    session_dir = store.session_dir
    assert session_dir.exists()

    store.close()
    assert not session_dir.exists()
