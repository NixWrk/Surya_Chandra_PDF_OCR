"""Minimal OCR GUI: input PDF + mode selection + progress + final PDF output."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from uniscan.app import (
    DEFAULT_BASIC_GUI_LANG,
    PDF_MODE_CHANDRA,
    PDF_MODE_HYBRID,
    PDF_MODE_SURYA,
    SearchablePdfSummary,
    build_searchable_pdf,
    parse_page_numbers,
)


DEFAULT_LANG = DEFAULT_BASIC_GUI_LANG
CPU_TORCH_PACKAGES: tuple[str, ...] = (
    "torch==2.11.0+cpu",
    "torchvision==0.26.0+cpu",
    "torchaudio==2.11.0+cpu",
)

MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Chandra + Surya (default)", PDF_MODE_HYBRID),
    ("Chandra", PDF_MODE_CHANDRA),
    ("Surya", PDF_MODE_SURYA),
)


class BasicOcrGui(tk.Tk):
    """Minimal launcher for searchable PDF generation."""

    def __init__(self) -> None:
        super().__init__()
        self.title("UniScan Basic OCR")
        self.geometry("820x280")
        self.minsize(760, 260)

        self.pdf_path_var = tk.StringVar()
        self.mode_label_var = tk.StringVar(value=MODE_OPTIONS[0][0])
        self.pages_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_text_var = tk.StringVar(value="0%")
        self.progress_var = tk.IntVar(value=0)
        self.delete_original_layer_var = tk.BooleanVar(value=False)

        self._worker: threading.Thread | None = None
        self._repair_worker: threading.Thread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=tk.BOTH, expand=True)

        row_file = ttk.Frame(root)
        row_file.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(row_file, text="PDF file:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row_file, textvariable=self.pdf_path_var).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(0, 8),
        )
        self.file_btn = ttk.Button(row_file, text="Browse", command=self._choose_pdf)
        self.file_btn.pack(side=tk.LEFT)

        row_mode = ttk.Frame(root)
        row_mode.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(row_mode, text="Mode:", width=12).pack(side=tk.LEFT)
        mode_labels = [label for label, _value in MODE_OPTIONS]
        self.mode_combo = ttk.Combobox(
            row_mode,
            values=mode_labels,
            textvariable=self.mode_label_var,
            state="readonly",
            width=28,
        )
        self.mode_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.mode_combo.current(0)
        ttk.Label(
            row_mode,
            text="Default: Chandra text + Surya geometry.",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        row_pages = ttk.Frame(root)
        row_pages.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(row_pages, text="Pages:", width=12).pack(side=tk.LEFT)
        self.pages_entry = ttk.Entry(row_pages, textvariable=self.pages_var)
        self.pages_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(row_pages, text="for example: 1,3,5-8 (blank = all)").pack(side=tk.LEFT)

        row_delete_layer = ttk.Frame(root)
        row_delete_layer.pack(fill=tk.X, pady=(0, 12))
        self.delete_layer_check = ttk.Checkbutton(
            row_delete_layer,
            variable=self.delete_original_layer_var,
            text="Remove existing text layer"
        )
        self.delete_layer_check.pack(side=tk.LEFT)

        progress_box = ttk.LabelFrame(root, text="Progress")
        progress_box.pack(fill=tk.X, pady=(0, 12))
        self.progress_bar = ttk.Progressbar(
            progress_box,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(10, 6))

        row_progress = ttk.Frame(progress_box)
        row_progress.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Label(row_progress, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row_progress, textvariable=self.progress_text_var, width=7, anchor="e").pack(side=tk.RIGHT)

        row_actions = ttk.Frame(root)
        row_actions.pack(fill=tk.X)
        self.start_btn = ttk.Button(row_actions, text="Run", command=self._start_run)
        self.start_btn.pack(side=tk.LEFT)

    def _choose_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.pdf_path_var.set(path)

    def _selected_mode(self) -> str:
        selected = self.mode_label_var.get().strip()
        for label, key in MODE_OPTIONS:
            if selected == label:
                return key
        return PDF_MODE_HYBRID

    def _set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.start_btn.configure(state=state)
        self.file_btn.configure(state=state)
        self.pages_entry.configure(state=state)
        self.mode_combo.configure(state="disabled" if running else "readonly")

    def _start_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        try:
            pdf_path = Path(self.pdf_path_var.get().strip())
            if not pdf_path.exists() or not pdf_path.is_file():
                raise RuntimeError("Choose an existing PDF file.")
            if pdf_path.suffix.lower() != ".pdf":
                raise RuntimeError("Only PDF input is supported.")

            mode = self._selected_mode()
            page_numbers = parse_page_numbers(self.pages_var.get())
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        self.progress_var.set(0)
        self.progress_text_var.set("0%")
        self.status_var.set("Preparing...")
        self._set_running(True)

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(pdf_path, mode, page_numbers, self.delete_original_layer_var.get()),
            daemon=True,
        )
        self._worker.start()

    def _run_worker(
        self,
        pdf_path: Path,
        mode: str,
        page_numbers: tuple[int, ...] | None,
        delete_original_layer: bool,
    ) -> None:
        try:
            summary = build_searchable_pdf(
                pdf_path=pdf_path,
                mode=mode,
                page_numbers=page_numbers,
                lang=DEFAULT_LANG,
                strict=True,
                overwrite_input_path=True,
                return_bytes=False,
                progress=self._queue_progress,
                delete_original_text_layer=delete_original_layer,
            )
            self.after(0, self._ui_done, summary)
        except Exception as exc:
            self.after(0, self._ui_error, str(exc))

    def _queue_progress(self, value: int, status: str) -> None:
        self.after(0, self._ui_set_progress, value, status)

    def _ui_set_progress(self, value: int, status: str) -> None:
        bounded = max(0, min(100, int(value)))
        self.progress_var.set(bounded)
        self.progress_text_var.set(f"{bounded}%")
        self.status_var.set(status)

    def _ui_done(self, summary: SearchablePdfSummary) -> None:
        self._set_running(False)
        self._ui_set_progress(100, "Completed")

        extra_lines: list[str] = []
        if summary.benchmark.skipped_engines:
            extra_lines.append("Skipped (missing dependencies):")
            extra_lines.extend(summary.benchmark.skipped_engines)
            extra_lines.append("")
        if summary.benchmark.failed_engines:
            extra_lines.append("Failed:")
            extra_lines.extend(summary.benchmark.failed_engines)
            extra_lines.append("")
        extra = ("\n".join(extra_lines)).strip()
        details_block = f"\n\n{extra}" if extra else ""

        messagebox.showinfo(
            "Done",
            "Searchable PDF has been built.\n\n"
            f"Mode: {summary.mode}\n"
            f"Output PDF:\n{summary.output_pdf_path}\n\n"
            f"Run artifacts:\n{summary.run_dir}"
            f"{details_block}",
        )

    def _ui_error(self, message: str) -> None:
        self._set_running(False)
        self.status_var.set("Error")
        if self._is_chandra_cuda_oom(message):
            self._offer_chandra_cpu_repair(message)
            return
        messagebox.showerror("Error", message)

    @staticmethod
    def _is_chandra_cuda_oom(message: str) -> bool:
        lowered = message.lower()
        if "chandra" not in lowered:
            return False
        return "cuda out of memory" in lowered or (
            "tried to allocate" in lowered and "gpu" in lowered
        )

    def _offer_chandra_cpu_repair(self, message: str) -> None:
        prompt = (
            "Chandra ran out of GPU memory.\n\n"
            "The current GPU does not have enough free VRAM for the Chandra model. "
            "Surya can still use the GPU, while Chandra can be switched to CPU mode.\n\n"
            "Do you want UniScan to install CPU PyTorch into the Chandra venv, "
            "switch Chandra to CPU mode, and restart the GUI?\n\n"
            "This can take several minutes and requires internet access."
        )
        if not messagebox.askyesno("Chandra GPU memory is not enough", prompt):
            messagebox.showerror(
                "Error",
                message
                + "\n\nTo retry manually, run with:\n"
                "$env:UNISCAN_CHANDRA_DEVICE_POLICY = \"cpu\"\n"
                ".\\run_basic_gui.cmd",
            )
            return
        self._start_chandra_cpu_repair()

    def _start_chandra_cpu_repair(self) -> None:
        if self._repair_worker is not None and self._repair_worker.is_alive():
            return
        self._set_running(True)
        self._ui_set_progress(0, "Installing Chandra CPU PyTorch...")
        self._repair_worker = threading.Thread(
            target=self._install_chandra_cpu_torch_worker,
            daemon=True,
        )
        self._repair_worker.start()

    def _install_chandra_cpu_torch_worker(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "--extra-index-url",
            "https://pypi.org/simple",
            "--upgrade",
            "--force-reinstall",
            *CPU_TORCH_PACKAGES,
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[3]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            self.after(0, self._ui_chandra_cpu_repair_failed, f"Failed to run pip: {exc}")
            return

        if proc.returncode != 0:
            details = (proc.stderr or proc.stdout or "").strip()
            if len(details) > 3000:
                details = details[-3000:]
            self.after(
                0,
                self._ui_chandra_cpu_repair_failed,
                details or f"pip exited with {proc.returncode}",
            )
            return

        self.after(0, self._ui_chandra_cpu_repair_done)

    def _ui_chandra_cpu_repair_failed(self, details: str) -> None:
        self._set_running(False)
        self.status_var.set("CPU PyTorch install failed")
        messagebox.showerror(
            "CPU PyTorch install failed",
            "UniScan could not install CPU PyTorch into the Chandra venv.\n\n"
            f"{details}",
        )

    def _ui_chandra_cpu_repair_done(self) -> None:
        os.environ["UNISCAN_CHANDRA_DEVICE_POLICY"] = "cpu"
        os.environ["TORCH_DEVICE"] = "cpu"
        os.environ["UNISCAN_CHANDRA_TORCH_DEVICE"] = "cpu"
        os.environ["UNISCAN_CHANDRA_PREFER_GPU"] = "0"
        os.environ["UNISCAN_CHANDRA_REQUIRE_GPU"] = "0"
        self._ui_set_progress(100, "Restarting in Chandra CPU mode...")
        messagebox.showinfo(
            "Chandra CPU mode installed",
            "CPU PyTorch was installed into the Chandra venv.\n\n"
            "UniScan will now restart with Chandra forced to CPU mode.",
        )
        os.execv(sys.executable, [sys.executable, "-m", "uniscan.ui.basic_ocr_gui"])


def main() -> int:
    app = BasicOcrGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
