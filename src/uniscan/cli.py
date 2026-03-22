"""CLI entrypoint for the unified scanner project."""

from __future__ import annotations

import argparse
from pathlib import Path

from uniscan.ocr import run_ocr_benchmark, summarize_ocr_benchmark
from uniscan.tools import run_crop_benchmark, summarize_benchmark_results
from uniscan.ui import run_app


def main(argv: list[str] | None = None) -> int:
    """Run unified scanner application."""
    parser = argparse.ArgumentParser(prog="uniscan")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    benchmark_parser = subparsers.add_parser(
        "benchmark-crop",
        help="Compare crop backends on one input folder and write one PDF per backend.",
    )
    benchmark_parser.add_argument("--input", required=True, type=Path, help="Input folder path.")
    benchmark_parser.add_argument("--output", required=True, type=Path, help="Output folder path.")
    benchmark_parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=300,
        help="Target DPI for generated PDFs.",
    )
    benchmark_parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help=(
            "Backend names to run. Defaults to paddleocr_uvdoc."
        ),
    )
    benchmark_parser.add_argument(
        "--scanner-root",
        type=Path,
        default=None,
        help="Optional root directory for vendored camscan backend.",
    )
    benchmark_parser.add_argument(
        "--uvdoc-cache",
        type=Path,
        default=None,
        help="Optional cache directory for PaddleOCR UVDoc weights.",
    )

    ocr_benchmark_parser = subparsers.add_parser(
        "benchmark-ocr",
        help="Run sampled OCR benchmarks on a PDF fixture and write engine outputs.",
    )
    ocr_benchmark_parser.add_argument("--pdf", required=True, type=Path, help="Input PDF fixture path.")
    ocr_benchmark_parser.add_argument("--output", required=True, type=Path, help="Output folder path.")
    ocr_benchmark_parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Total number of sampled pages (evenly distributed from first to last).",
    )
    ocr_benchmark_parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Render DPI for sampled pages.",
    )
    ocr_benchmark_parser.add_argument(
        "--lang",
        default="eng",
        help="OCR language code.",
    )
    ocr_benchmark_parser.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help=(
            "Engine names to run. Defaults to the registered OCR engine matrix."
        ),
    )
    ocr_benchmark_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when any engine is not ok.",
    )

    args = parser.parse_args(argv)
    if args.version:
        from uniscan import __version__

        print(__version__)
        return 0
    if args.command == "benchmark-crop":
        results = run_crop_benchmark(
            input_dir=args.input,
            output_dir=args.output,
            backends=tuple(args.backends) if args.backends else None,
            pdf_dpi=args.pdf_dpi,
            scanner_root=args.scanner_root,
            uvdoc_cache_home=args.uvdoc_cache,
        )
        print(summarize_benchmark_results(results))
        return 0 if any(result.output_pdf is not None for result in results) else 1
    if args.command == "benchmark-ocr":
        results = run_ocr_benchmark(
            pdf_path=args.pdf,
            output_dir=args.output,
            engines=tuple(args.engines) if args.engines else None,
            sample_size=args.sample_size,
            dpi=args.dpi,
            lang=args.lang,
        )
        print(summarize_ocr_benchmark(results))
        if args.strict and any(result.status != "ok" for result in results):
            return 1
        return 0
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
