"""Minimal OCR GUI: PDF file + mode selector + progress."""

from __future__ import annotations

import os
import subprocess
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from uniscan.ocr import detect_ocr_engine_status


DEFAULT_LANG = "rus+eng"
MODE_SURYA = "surya"
MODE_HYBRID = "hybrid"
MODE_BOTH = "both"

MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Surya", MODE_SURYA),
    ("Гибрид", MODE_HYBRID),
    ("Оба", MODE_BOTH),
)

MODE_TO_ENGINES: dict[str, tuple[str, ...]] = {
    MODE_SURYA: ("surya",),
    MODE_HYBRID: ("chandra",),
    MODE_BOTH: ("surya", "chandra"),
}


@dataclass(slots=True, frozen=True)
class RunSummary:
    run_dir: Path
    result_files: tuple[Path, ...]
    failed_engines: tuple[str, ...]
    skipped_engines: tuple[str, ...]


class BasicOcrGui(tk.Tk):
    """Minimal launcher for running OCR benchmark on selected engines."""

    def __init__(self) -> None:
        super().__init__()
        self.title("UniScan Basic OCR")
        self.geometry("720x250")
        self.minsize(680, 230)

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
            text="Гибрид = Chandra (с Surya-геометрией в текущем пайплайне)",
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

    def _parse_pages_spec(self, raw: str) -> tuple[int, ...] | None:
        normalized = raw.strip().replace("–", "-").replace("—", "-")
        if not normalized:
            return None

        tokens = [token for token in re.split(r"[,\s;]+", normalized) if token]
        if not tokens:
            return None

        pages: list[int] = []
        seen: set[int] = set()
        for token in tokens:
            if "-" in token:
                parts = token.split("-")
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(f"Некорректный диапазон: {token}")
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                except ValueError as exc:
                    raise ValueError(f"Некорректный диапазон: {token}") from exc
                if start < 1 or end < 1:
                    raise ValueError(f"Номер страницы должен быть >= 1: {token}")
                step = 1 if end >= start else -1
                for page in range(start, end + step, step):
                    if page in seen:
                        continue
                    seen.add(page)
                    pages.append(page)
                continue

            try:
                page = int(token)
            except ValueError as exc:
                raise ValueError(f"Некорректный номер страницы: {token}") from exc
            if page < 1:
                raise ValueError(f"Номер страницы должен быть >= 1: {page}")
            if page in seen:
                continue
            seen.add(page)
            pages.append(page)

        if not pages:
            return None
        return tuple(pages)

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
            requested_engines = MODE_TO_ENGINES.get(mode_key)
            if not requested_engines:
                raise RuntimeError("Не удалось определить выбранный режим.")
            page_numbers = self._parse_pages_spec(self.pages_var.get())

            ready_engines: list[str] = []
            skipped_engines: list[str] = []
            for engine in requested_engines:
                status = detect_ocr_engine_status(engine)
                if status.ready:
                    ready_engines.append(engine)
                    continue
                missing_deps = ", ".join(status.missing) if status.missing else "unknown"
                skipped_engines.append(f"{engine}: {missing_deps}")
            if not ready_engines:
                raise RuntimeError("Нет доступных движков:\n\n" + "\n".join(skipped_engines))
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))
            return

        self.progress_var.set(0)
        self.progress_text_var.set("0%")
        self.status_var.set("Подготовка...")
        self._set_running(True)

        if skipped_engines:
            self.status_var.set(
                "Часть движков пропущена: " + ", ".join(item.split(":", 1)[0] for item in skipped_engines)
            )

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(pdf_path, tuple(ready_engines), tuple(skipped_engines), page_numbers),
            daemon=True,
        )
        self._worker.start()

    def _run_worker(
        self,
        pdf_path: Path,
        engines: tuple[str, ...],
        skipped_engines: tuple[str, ...],
        page_numbers: tuple[int, ...] | None,
    ) -> None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = (Path.cwd() / "outputs" / "basic_gui_runs" / f"{pdf_path.stem}_{timestamp}").resolve()
            run_dir.mkdir(parents=True, exist_ok=True)

            result_files: list[Path] = []
            failed_engines: list[str] = []
            total = max(1, len(engines))

            for index, engine in enumerate(engines, start=1):
                start_percent = int(((index - 1) / total) * 100)
                end_percent = int((index / total) * 100)
                self.after(0, self._ui_set_progress, start_percent, f"Запуск: {engine}")

                engine_output = run_dir / engine
                engine_output.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-m",
                    "uniscan",
                    "benchmark-ocr",
                    "--pdf",
                    str(pdf_path),
                    "--output",
                    str(engine_output),
                    "--engines",
                    engine,
                    "--lang",
                    DEFAULT_LANG,
                    "--strict",
                ]
                if page_numbers is None:
                    cmd.extend(["--sample-size", "999999"])
                else:
                    cmd.extend(["--pages", ",".join(str(page) for page in page_numbers)])
                try:
                    self._run_engine_with_progress(
                        cmd=cmd,
                        engine=engine,
                        start_percent=start_percent,
                        end_percent=end_percent,
                    )
                except Exception as exc:
                    failed_engines.append(f"{engine}: {exc}")
                    self.after(0, self._ui_set_progress, end_percent, f"Ошибка: {engine}")
                    continue

                report_path = engine_output / f"{pdf_path.stem}_ocr_benchmark.json"
                result_files.append(report_path)
                self.after(0, self._ui_set_progress, end_percent, f"Готово: {engine}")

            if len(failed_engines) >= len(engines):
                details = "\n\n".join(failed_engines)
                raise RuntimeError(f"Ни один движок не завершился успешно.\n\n{details}")

            summary = RunSummary(
                run_dir=run_dir,
                result_files=tuple(result_files),
                failed_engines=tuple(failed_engines),
                skipped_engines=tuple(skipped_engines),
            )
            self.after(0, self._ui_done, summary)
        except Exception as exc:
            self.after(0, self._ui_error, str(exc))

    def _run_engine_with_progress(
        self,
        *,
        cmd: list[str],
        engine: str,
        start_percent: int,
        end_percent: int,
    ) -> None:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=self._engine_env(engine),
        )
        started = time.monotonic()
        span = max(1, end_percent - start_percent)
        pseudo = start_percent

        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.9)
                break
            except subprocess.TimeoutExpired:
                elapsed_seconds = int(time.monotonic() - started)
                advance = min(span - 1, elapsed_seconds // 3)
                target_progress = start_percent + advance
                if target_progress > pseudo:
                    pseudo = target_progress
                self.after(
                    0,
                    self._ui_set_progress,
                    pseudo,
                    f"Запуск: {engine} ({elapsed_seconds}с)",
                )

        if proc.returncode == 0:
            return

        tail = (stderr or stdout or "").strip()
        if tail:
            tail_lines = "\n".join(tail.splitlines()[-20:])
        else:
            tail_lines = "no error details"
        raise RuntimeError(f"Движок '{engine}' завершился с ошибкой:\n\n{tail_lines}")

    def _engine_env(self, engine: str) -> dict[str, str]:
        _ = engine
        return os.environ.copy()

    def _ui_set_progress(self, value: int, status: str) -> None:
        bounded = max(0, min(100, int(value)))
        self.progress_var.set(bounded)
        self.progress_text_var.set(f"{bounded}%")
        self.status_var.set(status)

    def _ui_done(self, summary: RunSummary) -> None:
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
