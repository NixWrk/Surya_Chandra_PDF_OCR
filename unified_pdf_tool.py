#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import img2pdf
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
ASCII_TMP_ROOT = Path(os.environ.get("SystemDrive", "C:")) / "_ocrmypdf_tmp"

PROFILE_SETTINGS = {
    "Fast": {"dpi": 220, "ocr_optimize": 3},
    "Balanced": {"dpi": 300, "ocr_optimize": 1},
    "Best quality": {"dpi": 400, "ocr_optimize": 0},
}


@contextlib.contextmanager
def ascii_tempdir(prefix: str = "ocrjob_"):
    ASCII_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=prefix, dir=str(ASCII_TMP_ROOT)))
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))

    if max_w < 10 or max_h < 10:
        return image

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_w, max_h))


def find_page_quad(image_bgr: np.ndarray):
    h, w = image_bgr.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale != 1.0 else image_bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 40, 160)
    edged = cv2.dilate(edged, None, iterations=1)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            if scale != 1.0:
                quad = (quad / scale).astype(np.float32)
            return quad

    return None


def deskew_and_crop(image_bgr: np.ndarray, crop_pad: int = 12) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    coords = cv2.findNonZero(thr)
    if coords is None:
        return image_bgr

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle

    h, w = image_bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_bgr,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thr2 = cv2.adaptiveThreshold(
        gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    thr2 = cv2.morphologyEx(thr2, cv2.MORPH_OPEN, kernel, iterations=1)
    coords2 = cv2.findNonZero(thr2)
    if coords2 is None:
        return rotated

    x, y, ww, hh = cv2.boundingRect(coords2)
    x = max(0, x - crop_pad)
    y = max(0, y - crop_pad)
    ww = min(rotated.shape[1] - x, ww + 2 * crop_pad)
    hh = min(rotated.shape[0] - y, hh + 2 * crop_pad)
    return rotated[y : y + hh, x : x + ww]


def split_spread(image_bgr: np.ndarray):
    _, w = image_bgr.shape[:2]
    mid = w // 2
    return image_bgr[:, :mid], image_bgr[:, mid:]


def preprocess_image(path: Path, do_perspective: bool, do_deskew_crop: bool, split: bool):
    img = imread_unicode(path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    if do_perspective:
        quad = find_page_quad(img)
        if quad is not None:
            img = four_point_transform(img, quad)

    pages = split_spread(img) if split else (img,)
    output = []
    for page in pages:
        if do_deskew_crop:
            page = deskew_and_crop(page)
        output.append(page)
    return output


def build_pdf_from_images(image_paths: list[Path], out_pdf: Path, dpi: int):
    with open(out_pdf, "wb") as f:
        try:
            payload = img2pdf.convert([str(p) for p in image_paths], dpi=dpi)
        except TypeError:
            layout = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
            payload = img2pdf.convert([str(p) for p in image_paths], layout_fun=layout)
        f.write(payload)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unified PDF Tool (images folder / file -> PDF, optional OCR)")
        self.geometry("900x560")
        self.minsize(840, 520)

        self.events = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None

        self.mode = tk.StringVar(value="images_folder")
        self.in_dir = tk.StringVar()
        self.in_file = tk.StringVar()
        self.out_pdf = tk.StringVar()

        self.profile = tk.StringVar(value="Balanced")
        self.enable_preprocess = tk.BooleanVar(value=True)
        self.perspective = tk.BooleanVar(value=True)
        self.deskew_crop = tk.BooleanVar(value=True)
        self.split_spreads = tk.BooleanVar(value=False)

        self.enable_ocr = tk.BooleanVar(value=True)
        self.lang = tk.StringVar(value="rus+eng")
        self.skip_text_pdf = tk.BooleanVar(value=True)

        self.stage = tk.StringVar(value="Idle")
        self.current = tk.StringVar(value="-")
        self.percent = tk.StringVar(value="0%")

        self._build_ui()
        self._apply_mode_ui()
        self._apply_toggles()
        self.after(120, self._poll_events)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        mode_box = ttk.LabelFrame(root, text="Input mode")
        mode_box.pack(fill="x", **pad)
        mode_row = ttk.Frame(mode_box)
        mode_row.pack(fill="x", padx=10, pady=8)
        ttk.Radiobutton(
            mode_row,
            text="Images folder",
            variable=self.mode,
            value="images_folder",
            command=self._apply_mode_ui,
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            mode_row,
            text="Single file (image or PDF)",
            variable=self.mode,
            value="single_file",
            command=self._apply_mode_ui,
        ).pack(side="left", padx=8)

        r1 = ttk.Frame(root)
        r1.pack(fill="x", **pad)
        ttk.Label(r1, text="Images folder:").pack(side="left")
        self.in_dir_entry = ttk.Entry(r1, textvariable=self.in_dir)
        self.in_dir_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.in_dir_btn = ttk.Button(r1, text="Choose...", command=self.choose_in_dir)
        self.in_dir_btn.pack(side="left")

        r2 = ttk.Frame(root)
        r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Input file:").pack(side="left")
        self.in_file_entry = ttk.Entry(r2, textvariable=self.in_file)
        self.in_file_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.in_file_btn = ttk.Button(r2, text="Choose...", command=self.choose_in_file)
        self.in_file_btn.pack(side="left")

        r3 = ttk.Frame(root)
        r3.pack(fill="x", **pad)
        ttk.Label(r3, text="Save PDF as:").pack(side="left")
        ttk.Entry(r3, textvariable=self.out_pdf).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r3, text="Save as...", command=self.choose_out_pdf).pack(side="left")

        opts = ttk.LabelFrame(root, text="Options")
        opts.pack(fill="x", **pad)

        o1 = ttk.Frame(opts)
        o1.pack(fill="x", padx=10, pady=6)
        ttk.Label(o1, text="Profile:").pack(side="left")
        ttk.Combobox(
            o1,
            textvariable=self.profile,
            values=list(PROFILE_SETTINGS.keys()),
            width=14,
            state="readonly",
        ).pack(side="left", padx=8)
        ttk.Label(o1, text="Fast: smaller/faster, Balanced: default, Best quality: larger/slower").pack(
            side="left", padx=8
        )

        o2 = ttk.Frame(opts)
        o2.pack(fill="x", padx=10, pady=6)
        self.preprocess_chk = ttk.Checkbutton(
            o2, text="Preprocess images", variable=self.enable_preprocess, command=self._apply_toggles
        )
        self.preprocess_chk.pack(side="left")
        self.perspective_chk = ttk.Checkbutton(o2, text="Auto perspective", variable=self.perspective)
        self.perspective_chk.pack(side="left", padx=12)
        self.deskew_chk = ttk.Checkbutton(o2, text="Auto deskew + crop", variable=self.deskew_crop)
        self.deskew_chk.pack(side="left", padx=12)
        self.split_chk = ttk.Checkbutton(o2, text="Split spreads", variable=self.split_spreads)
        self.split_chk.pack(side="left", padx=12)

        o3 = ttk.Frame(opts)
        o3.pack(fill="x", padx=10, pady=6)
        self.ocr_chk = ttk.Checkbutton(o3, text="Enable OCR (ocrmypdf)", variable=self.enable_ocr, command=self._apply_toggles)
        self.ocr_chk.pack(side="left")
        ttk.Label(o3, text="Lang:").pack(side="left", padx=(14, 0))
        self.lang_entry = ttk.Entry(o3, textvariable=self.lang, width=12)
        self.lang_entry.pack(side="left", padx=6)
        self.skip_text_chk = ttk.Checkbutton(
            o3, text="For PDF input: skip pages that already have text", variable=self.skip_text_pdf
        )
        self.skip_text_chk.pack(side="left", padx=12)

        prog = ttk.LabelFrame(root, text="Progress")
        prog.pack(fill="x", **pad)

        p1 = ttk.Frame(prog)
        p1.pack(fill="x", padx=10, pady=6)
        ttk.Label(p1, text="Stage:").pack(side="left")
        ttk.Label(p1, textvariable=self.stage).pack(side="left", padx=8)

        p2 = ttk.Frame(prog)
        p2.pack(fill="x", padx=10, pady=6)
        ttk.Label(p2, text="Current:").pack(side="left")
        ttk.Label(p2, textvariable=self.current).pack(side="left", padx=8)

        p3 = ttk.Frame(prog)
        p3.pack(fill="x", padx=10, pady=6)
        self.bar = ttk.Progressbar(p3, orient="horizontal", mode="determinate", maximum=100)
        self.bar.pack(side="left", fill="x", expand=True)
        ttk.Label(p3, textvariable=self.percent, width=8, anchor="e").pack(side="left", padx=8)

        btns = ttk.Frame(root)
        btns.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=10)

    def _apply_mode_ui(self):
        is_folder = self.mode.get() == "images_folder"
        self.in_dir_entry.configure(state="normal" if is_folder else "disabled")
        self.in_dir_btn.configure(state="normal" if is_folder else "disabled")
        self.in_file_entry.configure(state="disabled" if is_folder else "normal")
        self.in_file_btn.configure(state="disabled" if is_folder else "normal")

    def _apply_toggles(self):
        preprocess_state = "normal" if self.enable_preprocess.get() else "disabled"
        self.perspective_chk.configure(state=preprocess_state)
        self.deskew_chk.configure(state=preprocess_state)
        self.split_chk.configure(state=preprocess_state)

        ocr_state = "normal" if self.enable_ocr.get() else "disabled"
        self.lang_entry.configure(state=ocr_state)
        self.skip_text_chk.configure(state=ocr_state)

    def choose_in_dir(self):
        d = filedialog.askdirectory(title="Select folder with images")
        if d:
            self.in_dir.set(d)

    def choose_in_file(self):
        f = filedialog.askopenfilename(
            title="Select image or PDF",
            filetypes=[
                ("Images and PDF", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.webp;*.bmp;*.pdf"),
                ("PDF files", "*.pdf"),
                ("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.webp;*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if f:
            self.in_file.set(f)

    def choose_out_pdf(self):
        f = filedialog.asksaveasfilename(
            title="Save PDF as",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if f:
            self.out_pdf.set(f)

    def set_busy(self, busy: bool):
        self.start_btn.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if busy else "disabled")

    def start(self):
        try:
            job = self._build_job()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if self.enable_ocr.get() and not which("ocrmypdf"):
            messagebox.showerror(
                "Error",
                "OCR is enabled, but 'ocrmypdf' is not found in PATH.\nInstall ocrmypdf or disable OCR.",
            )
            return

        self.stop_flag.clear()
        self.stage.set("Starting...")
        self.current.set("-")
        self.percent.set("0%")
        self.bar.stop()
        self.bar.configure(mode="determinate")
        self.bar["value"] = 0
        self.set_busy(True)

        self.worker = threading.Thread(target=self._worker, args=(job,), daemon=True)
        self.worker.start()

    def cancel(self):
        self.stop_flag.set()
        self.stage.set("Cancelling...")

    def _build_job(self):
        out_pdf = Path(self.out_pdf.get().strip())
        if not out_pdf.name:
            raise RuntimeError("Please choose a valid output PDF path.")
        if out_pdf.suffix.lower() != ".pdf":
            out_pdf = out_pdf.with_suffix(".pdf")
            self.out_pdf.set(str(out_pdf))
        out_pdf.parent.mkdir(parents=True, exist_ok=True)

        if self.mode.get() == "images_folder":
            in_dir = Path(self.in_dir.get().strip())
            if not in_dir.exists() or not in_dir.is_dir():
                raise RuntimeError("Please choose a valid images folder.")

            images = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            images.sort(key=lambda p: natural_key(p.name))
            if not images:
                raise RuntimeError("No supported images found in the selected folder.")

            return {"kind": "images", "images": images, "out_pdf": out_pdf}

        in_file = Path(self.in_file.get().strip())
        if not in_file.exists() or not in_file.is_file():
            raise RuntimeError("Please choose a valid input file.")

        ext = in_file.suffix.lower()
        if ext in IMG_EXTS:
            return {"kind": "images", "images": [in_file], "out_pdf": out_pdf}
        if ext == ".pdf":
            return {"kind": "pdf", "pdf": in_file, "out_pdf": out_pdf}

        raise RuntimeError("Supported inputs are image files and PDF.")

    def _emit(self, kind, payload=None):
        self.events.put((kind, payload))

    def _prepare_images(
        self,
        images: list[Path],
        out_dir: Path,
        do_preprocess: bool,
        do_perspective: bool,
        do_deskew_crop: bool,
        split: bool,
    ) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        output = []
        counter = 1
        total = len(images)

        self._emit("stage", ("Preparing images...", "determinate"))

        for idx, src in enumerate(images, start=1):
            if self.stop_flag.is_set():
                raise RuntimeError("Cancelled by user.")

            self._emit("current", src.name)

            if do_preprocess:
                pages = preprocess_image(
                    src,
                    do_perspective=do_perspective,
                    do_deskew_crop=do_deskew_crop,
                    split=split,
                )
            else:
                img = imread_unicode(src)
                if img is None:
                    raise RuntimeError(f"Cannot read image: {src}")
                pages = [img]

            for page in pages:
                out_path = out_dir / f"{counter:05d}.png"
                if not imwrite_unicode(out_path, page):
                    raise RuntimeError(f"Failed to write image: {out_path}")
                output.append(out_path)
                counter += 1

            self._emit("progress", int((idx / total) * 100))

        return output

    def _run_ocr(self, in_pdf: Path, out_pdf: Path, lang: str, optimize: int, skip_text_for_pdf: bool):
        jobs = max(1, min(8, (os.cpu_count() or 2) - 1))
        cmd = [
            "ocrmypdf",
            str(in_pdf),
            str(out_pdf),
            "-l",
            lang,
            "-O",
            str(optimize),
            "--rotate-pages",
            "--jobs",
            str(jobs),
        ]
        if skip_text_for_pdf:
            cmd.append("--skip-text")

        self._emit("stage", ("Running OCR (ocrmypdf)...", "indeterminate"))
        self._emit("current", "OCR in progress...")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines = []
        try:
            while True:
                if self.stop_flag.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                    raise RuntimeError("Cancelled by user.")

                if proc.poll() is not None:
                    break
                time.sleep(0.2)
        finally:
            if proc.stdout:
                for ln in proc.stdout.read().splitlines():
                    ln = ln.strip()
                    if ln:
                        lines.append(ln)
                        m = re.search(r"(\d{1,3})%", ln)
                        if m:
                            self._emit("progress", int(m.group(1)))
                lines = lines[-20:]

        if proc.returncode != 0:
            tail = "\n".join(lines[-12:])
            msg = "ocrmypdf failed."
            if tail:
                msg += "\n\n" + tail
            raise RuntimeError(msg)

    def _worker(self, job):
        try:
            profile = PROFILE_SETTINGS[self.profile.get()]
            dpi = int(profile["dpi"])
            optimize = int(profile["ocr_optimize"])

            do_preprocess = bool(self.enable_preprocess.get())
            do_perspective = bool(self.perspective.get())
            do_deskew_crop = bool(self.deskew_crop.get())
            split = bool(self.split_spreads.get())

            do_ocr = bool(self.enable_ocr.get())
            lang = self.lang.get().strip() or "rus+eng"
            skip_text_for_pdf = bool(self.skip_text_pdf.get())

            out_pdf: Path = job["out_pdf"]

            with ascii_tempdir(prefix="unified_pdf_") as tmp_root:
                raw_pdf = tmp_root / "raw.pdf"

                if job["kind"] == "images":
                    prepared_dir = tmp_root / "prepared"
                    prepared_images = self._prepare_images(
                        images=job["images"],
                        out_dir=prepared_dir,
                        do_preprocess=do_preprocess,
                        do_perspective=do_perspective,
                        do_deskew_crop=do_deskew_crop,
                        split=split,
                    )

                    if self.stop_flag.is_set():
                        raise RuntimeError("Cancelled by user.")

                    self._emit("stage", ("Building PDF...", "indeterminate"))
                    self._emit("current", f"Packing {len(prepared_images)} page(s)")
                    build_pdf_from_images(prepared_images, raw_pdf, dpi=dpi)

                else:
                    in_pdf: Path = job["pdf"]
                    self._emit("stage", ("Preparing input PDF...", "indeterminate"))
                    self._emit("current", in_pdf.name)
                    shutil.copy2(in_pdf, raw_pdf)

                if self.stop_flag.is_set():
                    raise RuntimeError("Cancelled by user.")

                if do_ocr:
                    self._run_ocr(
                        in_pdf=raw_pdf,
                        out_pdf=out_pdf,
                        lang=lang,
                        optimize=optimize,
                        skip_text_for_pdf=(job["kind"] == "pdf" and skip_text_for_pdf),
                    )
                else:
                    self._emit("stage", ("Saving PDF (OCR disabled)...", "indeterminate"))
                    self._emit("current", out_pdf.name)
                    shutil.copy2(raw_pdf, out_pdf)

            self._emit("done", str(out_pdf))
        except Exception as e:
            self._emit("error", str(e))

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()

                if kind == "stage":
                    text, mode = payload
                    self.stage.set(text)
                    if mode == "indeterminate":
                        self.bar.configure(mode="indeterminate")
                        self.bar.start(10)
                        self.percent.set("...")
                    else:
                        self.bar.stop()
                        self.bar.configure(mode="determinate")
                elif kind == "current":
                    self.current.set(payload)
                elif kind == "progress":
                    if str(self.bar.cget("mode")) == "indeterminate":
                        self.bar.stop()
                        self.bar.configure(mode="determinate")
                    v = max(0, min(100, int(payload)))
                    self.bar["value"] = v
                    self.percent.set(f"{v}%")
                elif kind == "done":
                    self.bar.stop()
                    self.bar.configure(mode="determinate")
                    self.bar["value"] = 100
                    self.percent.set("100%")
                    self.stage.set("Done")
                    self.current.set(payload)
                    self.set_busy(False)
                    messagebox.showinfo("Done", f"Saved:\n{payload}")
                elif kind == "error":
                    self.bar.stop()
                    self.stage.set("Error")
                    self.set_busy(False)
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        finally:
            self.after(120, self._poll_events)


if __name__ == "__main__":
    App().mainloop()
