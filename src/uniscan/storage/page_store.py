"""Disk-backed page storage to keep memory usage low."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from uniscan.io.loaders import imwrite_unicode


class PageStore:
    """Manage per-session page files (original/current/thumbnail)."""

    def __init__(self, root_dir: Path | None = None, *, keep_on_close: bool = False) -> None:
        base = Path(root_dir) if root_dir is not None else Path(tempfile.gettempdir()) / "uniscan_cache"
        self.session_id = uuid4().hex
        self.session_dir = base / self.session_id
        self.pages_dir = self.session_dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.keep_on_close = keep_on_close

    def paths_for_entry(self, entry_id: str) -> tuple[Path, Path, Path]:
        page_dir = self.pages_dir / entry_id
        page_dir.mkdir(parents=True, exist_ok=True)
        original = page_dir / "original.png"
        current = page_dir / "current.png"
        thumb = page_dir / "thumb.jpg"
        return original, current, thumb

    def read_image(self, path: Path) -> np.ndarray:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Cannot read page image: {path}")
        return image

    def write_image(self, path: Path, image: np.ndarray) -> None:
        if not imwrite_unicode(path, image):
            raise RuntimeError(f"Cannot write page image: {path}")

    def write_thumbnail(self, path: Path, image: np.ndarray, *, max_side: int = 320) -> None:
        if len(image.shape) == 2:
            preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            preview = image
        h, w = preview.shape[:2]
        scale = min(max_side / max(1, w), max_side / max(1, h), 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        thumb = cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if not imwrite_unicode(path, thumb):
            raise RuntimeError(f"Cannot write page thumbnail: {path}")

    def add_page(self, entry_id: str, image: np.ndarray) -> tuple[Path, Path, Path]:
        original_path, current_path, thumb_path = self.paths_for_entry(entry_id)
        self.write_image(original_path, image)
        self.write_image(current_path, image)
        self.write_thumbnail(thumb_path, image)
        return original_path, current_path, thumb_path

    def remove_page(self, entry_id: str) -> None:
        page_dir = self.pages_dir / entry_id
        shutil.rmtree(page_dir, ignore_errors=True)

    def close(self) -> None:
        if self.keep_on_close:
            return
        shutil.rmtree(self.session_dir, ignore_errors=True)
