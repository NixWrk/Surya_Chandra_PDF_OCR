"""Minimal OCR GUI: PDF file + mode selector + progress."""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from uniscan.app import (
    DEFAULT_BASIC_GUI_LANG,
    MODE_BOTH,
    MODE_HYBRID,
    MODE_SURYA,
    BasicOcrRunSummary,
    MODE_TO_ENGINES,
    parse_page_numbers,
    run_basic_ocr_benchmark,
)


DEFAULT_LANG = DEFAULT_BASIC_GUI_LANG

MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Surya", MODE_SURYA),
    ("Гибрид", MODE_HYBRID),
    ("Оба", MODE_BOTH),
)


class BasicOcrGui(tk.Tk):
    """Minimal launcher for running OCR benchmark on selected engines."""

    def __init__(self) -> None:
        super().__init__()
        self.title("UniScan Basic OCR")
        self.geometry("760x260")
        self.minsize(700, 240)

        self.pdf_path_var = tk.StringVar()
        self.mode_label_var = tk.StringVar(value=MODE_OPTIONS[0][0])
        self.pages_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Готово")
        self.progress_text_var = tk.StringVar(value="0%")
        self.progress_var = tk.IntVar(value=0)

        self._worker: threading.Thread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=tk.BOTH, expand=True)

        row_file = ttk.Frame(root)
        row_file.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(row_file, text="PDF файл:", width=12).pack(side=tk.LEFT)
        ttk.Entry(row_file, textvariable=self.pdf_path_var).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(0, 8),
        )
        self.file_btn = ttk.Button(row_file, text="Выбрать", command=self._choose_pdf)
        self.file_btn.pack(side=tk.LEFT)

        row_mode = ttk.Frame(root)
        row_mode.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(row_mode, text="Модель:", width=12).pack(side=tk.LEFT)
        mode_labels = [label for label, _value in MODE_OPTIONS]
        self.mode_combo = ttk.Combobox(
            row_mode,
            values=mode_labels,
            textvariable=self.mode_label_var,
            state="readonly",
            width=20,
        )
        self.mode_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.mode_combo.current(0)
        ttk.Label(
            row_mode,
            text="Гибрид = Chandra (совмещение с доступной геометрией в пайплайне)",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        row_pages = ttk.Frame(root)
        row_pages.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(row_pages, text="Страницы:", width=12).pack(side=tk.LEFT)
        self.pages_entry = ttk.Entry(row_pages, textvariable=self.pages_var)
        self.pages_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(row_pages, text="напр.: 1,3,5-8 (пусто = все)").pack(side=tk.LEFT)

        progress_box = ttk.LabelFrame(root, text="Прогресс")
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
        self.start_btn = ttk.Button(row_actions, text="Запустить", command=self._start_run)
        self.start_btn.pack(side=tk.LEFT)

    def _choose_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Выбери PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.pdf_path_var.set(path)

    def _selected_mode_key(self) -> str:
        selected = self.mode_label_var.get().strip()
        for label, key in MODE_OPTIONS:
            if selected == label:
                return key
        return MODE_SURYA

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
                raise RuntimeError("Выбери существующий PDF файл.")
            if pdf_path.suffix.lower() != ".pdf":
                raise RuntimeError("Поддерживается только PDF.")

            mode_key = self._selected_mode_key()
            if mode_key not in MODE_TO_ENGINES:
                raise RuntimeError("Не удалось определить выбранный режим.")
            page_numbers = parse_page_numbers(self.pages_var.get())
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))
            return

        self.progress_var.set(0)
        self.progress_text_var.set("0%")
        self.status_var.set("Подготовка...")
        self._set_running(True)

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(pdf_path, mode_key, page_numbers),
            daemon=True,
        )
        self._worker.start()

    def _run_worker(
        self,
        pdf_path: Path,
        mode_key: str,
        page_numbers: tuple[int, ...] | None,
    ) -> None:
        try:
            summary = run_basic_ocr_benchmark(
                pdf_path=pdf_path,
                mode_key=mode_key,
                page_numbers=page_numbers,
                lang=DEFAULT_LANG,
                progress=self._queue_progress,
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

    def _ui_done(self, summary: BasicOcrRunSummary) -> None:
        self._set_running(False)
        self._ui_set_progress(100, "Завершено")
        result_lines = "\n".join(str(path) for path in summary.result_files if path.exists())
        if not result_lines:
            result_lines = "(файлы отчётов не найдены)"

        extra_lines: list[str] = []
        if summary.skipped_engines:
            extra_lines.append("Пропущены (нет зависимостей):")
            extra_lines.extend(summary.skipped_engines)
            extra_lines.append("")
        if summary.failed_engines:
            extra_lines.append("С ошибкой:")
            extra_lines.extend(summary.failed_engines)
            extra_lines.append("")
        extra = ("\n".join(extra_lines)).strip()
        details_block = f"\n\n{extra}" if extra else ""

        messagebox.showinfo(
            "Готово",
            "OCR выполнен.\n\n"
            f"Папка результата:\n{summary.run_dir}\n\n"
            f"Отчёты:\n{result_lines}"
            f"{details_block}",
        )

    def _ui_error(self, message: str) -> None:
        self._set_running(False)
        self.status_var.set("Ошибка")
        messagebox.showerror("Ошибка", message)


def main() -> int:
    app = BasicOcrGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

