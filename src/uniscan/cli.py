"""CLI entrypoint for the unified scanner project."""

from __future__ import annotations

import argparse
from pathlib import Path

from uniscan.app import (
    PDF_MODE_CHANDRA,
    PDF_MODE_HYBRID,
    PDF_MODE_SURYA,
    build_chandra_geometry_variants,
    build_searchable_pdf,
    parse_page_numbers,
)
from uniscan.ocr import (
    build_compare_txt_from_benchmark,
    run_artifact_searchable_package,
    run_ocr_benchmark,
    run_ocr_canonical_package,
    summarize_compare_txt_build,
    summarize_artifact_searchable_package,
    summarize_ocr_benchmark,
    summarize_ocr_canonical_package,
)
from uniscan.ocr.preprocessing import PREPROCESSING_MODES
from uniscan.tools import run_crop_benchmark, summarize_benchmark_results
from uniscan.ui import run_app
from uniscan.web import run_http_server


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
        "--pages",
        nargs="+",
        default=None,
        help="Explicit 1-based pages (for example: --pages 3,9). Overrides --sample-size.",
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

    ocr_canonical_parser = subparsers.add_parser(
        "benchmark-ocr-canonical",
        help="Build equal-condition OCR package (canonical per-page text + normalized searchable PDFs).",
    )
    ocr_canonical_parser.add_argument("--pdf", required=True, type=Path, help="Input PDF fixture path.")
    ocr_canonical_parser.add_argument("--output", required=True, type=Path, help="Output folder path.")
    ocr_canonical_parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Total number of sampled pages (evenly distributed from first to last).",
    )
    ocr_canonical_parser.add_argument(
        "--pages",
        nargs="+",
        default=None,
        help="Explicit 1-based pages (for example: --pages 3,9). Overrides --sample-size.",
    )
    ocr_canonical_parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Render DPI for sampled pages.",
    )
    ocr_canonical_parser.add_argument(
        "--lang",
        default="eng",
        help="OCR language code.",
    )
    ocr_canonical_parser.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help="Engine names to run. Defaults to the registered OCR engine matrix.",
    )
    ocr_canonical_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when any engine is not ok.",
    )
    ocr_canonical_parser.add_argument(
        "--render-dpi",
        type=int,
        default=0,
        help=(
            "DPI for rendering PDF pages to images. "
            "When > 0 overrides --dpi for the render step. "
            "Default 0 falls back to --dpi."
        ),
    )
    ocr_canonical_parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=0,
        help=(
            "DPI at which images are fed to the OCR engine. "
            "When render-dpi != ocr-dpi images are rescaled before OCR. "
            "Default 0 means same as render-dpi."
        ),
    )
    ocr_canonical_parser.add_argument(
        "--preprocessing",
        choices=list(PREPROCESSING_MODES),
        default="none",
        help=(
            "Pre-processing applied to each page image before OCR: "
            "'none' (default) — pass through unchanged; "
            "'basic' — greyscale + DPI normalise; "
            "'full' — basic + Otsu binarise + deskew."
        ),
    )

    artifact_searchable_parser = subparsers.add_parser(
        "build-searchable-from-artifacts",
        help="Build searchable PDFs from existing OCR TXT artifacts (artifact-first mode).",
    )
    artifact_searchable_parser.add_argument(
        "--compare-dir",
        required=True,
        type=Path,
        help="Folder with '<document>__<engine>.txt' files.",
    )
    artifact_searchable_parser.add_argument(
        "--pdf-root",
        required=True,
        type=Path,
        help="Root folder with source PDFs.",
    )
    artifact_searchable_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output folder for searchable PDFs and summary files.",
    )
    artifact_searchable_parser.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help="Engine names to include (for example: chandra surya olmocr).",
    )
    artifact_searchable_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when any artifact conversion is not ok.",
    )
    artifact_searchable_parser.add_argument(
        "--require-page-markers",
        action="store_true",
        help="Require explicit page markers in TXT artifacts ([SOURCE PAGE N] or form-feed).",
    )

    compare_prepare_parser = subparsers.add_parser(
        "prepare-compare-txt",
        help="Build '<document>__<engine>.txt' compare folder from benchmark summary/artifacts.",
    )
    compare_prepare_parser.add_argument(
        "--benchmark-root",
        required=True,
        type=Path,
        help="Benchmark root folder containing summary.json and engine artifacts.",
    )
    compare_prepare_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output compare_txt folder.",
    )
    compare_prepare_parser.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help="Optional engine filter (for example: chandra surya olmocr).",
    )
    compare_prepare_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when any selected engine failed to export compare_txt.",
    )

    geometry_compare_parser = subparsers.add_parser(
        "compare-chandra-geometry",
        help="Build two searchable PDFs: chandra text + (chandra geometry / surya geometry).",
    )
    geometry_compare_parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="Benchmark run root folder (contains chandra/surya artifacts).",
    )
    geometry_compare_parser.add_argument(
        "--pdf-root",
        required=True,
        type=Path,
        help="Root folder with source PDFs.",
    )
    geometry_compare_parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Optional output root for geometry-compare PDFs.",
    )
    geometry_compare_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when any stage failed.",
    )

    searchable_pdf_parser = subparsers.add_parser(
        "searchable-pdf",
        help=(
            "Process one PDF and return one searchable PDF. "
            "By default runs hybrid chandra+surya mode and overwrites input path."
        ),
    )
    searchable_pdf_parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Input PDF path. The file is overwritten with searchable output.",
    )
    searchable_pdf_parser.add_argument(
        "--mode",
        choices=[PDF_MODE_CHANDRA, PDF_MODE_SURYA, PDF_MODE_HYBRID],
        default=PDF_MODE_HYBRID,
        help="OCR mode. Default: chandra+surya.",
    )
    searchable_pdf_parser.add_argument(
        "--pages",
        nargs="+",
        default=None,
        help="Optional 1-based pages (for example: --pages 1,3,5-8).",
    )
    searchable_pdf_parser.add_argument(
        "--lang",
        default="rus+eng",
        help="OCR language code passed to benchmark stage.",
    )
    searchable_pdf_parser.add_argument(
        "--work-root",
        default=None,
        type=Path,
        help="Optional workspace for intermediate run artifacts.",
    )
    searchable_pdf_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code on any failed stage.",
    )

    serve_http_parser = subparsers.add_parser(
        "serve-http",
        help="Run minimal HTTP API (PDF bytes in -> PDF bytes out).",
    )
    serve_http_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/IP to bind HTTP server to.",
    )
    serve_http_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port for HTTP server.",
    )
    serve_http_parser.add_argument(
        "--work-root",
        default=None,
        type=Path,
        help="Optional workspace for intermediate run artifacts.",
    )
    serve_http_parser.add_argument(
        "--lang",
        default="rus+eng",
        help="Default OCR language code for incoming HTTP requests.",
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
        try:
            page_numbers = parse_page_numbers(args.pages)
        except ValueError as exc:
            parser.error(str(exc))
        results = run_ocr_benchmark(
            pdf_path=args.pdf,
            output_dir=args.output,
            engines=tuple(args.engines) if args.engines else None,
            sample_size=args.sample_size,
            page_numbers=page_numbers,
            dpi=args.dpi,
            lang=args.lang,
        )
        print(summarize_ocr_benchmark(results))
        if args.strict and any(result.status != "ok" for result in results):
            return 1
        return 0
    if args.command == "benchmark-ocr-canonical":
        try:
            page_numbers = parse_page_numbers(args.pages)
        except ValueError as exc:
            parser.error(str(exc))
        results = run_ocr_canonical_package(
            pdf_path=args.pdf,
            output_dir=args.output,
            engines=tuple(args.engines) if args.engines else None,
            sample_size=args.sample_size,
            page_numbers=page_numbers,
            dpi=args.dpi,
            render_dpi=args.render_dpi,
            ocr_dpi=args.ocr_dpi,
            preprocessing=args.preprocessing,
            lang=args.lang,
        )
        print(summarize_ocr_canonical_package(results))
        if args.strict and any(result.status != "ok" for result in results):
            return 1
        return 0
    if args.command == "build-searchable-from-artifacts":
        results = run_artifact_searchable_package(
            compare_dir=args.compare_dir,
            pdf_root=args.pdf_root,
            output_dir=args.output,
            engines=tuple(args.engines) if args.engines else None,
            require_page_markers=bool(args.require_page_markers or args.strict),
        )
        print(summarize_artifact_searchable_package(results))
        if args.strict and any(result.status != "ok" for result in results):
            return 1
        return 0
    if args.command == "prepare-compare-txt":
        results = build_compare_txt_from_benchmark(
            benchmark_root=args.benchmark_root,
            output_dir=args.output,
            engines=tuple(args.engines) if args.engines else None,
        )
        print(summarize_compare_txt_build(results))
        if args.strict and any(result.status != "ok" for result in results):
            return 1
        return 0
    if args.command == "compare-chandra-geometry":
        summary = build_chandra_geometry_variants(
            run_root=args.run_root,
            pdf_root=args.pdf_root,
            output_root=args.output,
            strict=bool(args.strict),
        )
        print(f"compare_dir={summary.compare_dir}")
        print(f"output_root={summary.output_root}")
        print("variants:")
        print("  - chandra_text__chandra_geometry")
        print("  - chandra_text__surya_geometry")
        return 0
    if args.command == "searchable-pdf":
        try:
            page_numbers = parse_page_numbers(args.pages)
        except ValueError as exc:
            parser.error(str(exc))
        summary = build_searchable_pdf(
            pdf_path=args.pdf,
            mode=args.mode,
            page_numbers=page_numbers,
            lang=args.lang,
            work_root=args.work_root,
            strict=bool(args.strict),
            overwrite_input_path=True,
            return_bytes=False,
        )
        print(f"mode={summary.mode}")
        print(f"run_dir={summary.run_dir}")
        print(f"output_pdf={summary.output_pdf_path}")
        if summary.overwritten_input_path is not None:
            print(f"overwritten={summary.overwritten_input_path}")
        return 0
    if args.command == "serve-http":
        run_http_server(
            host=args.host,
            port=int(args.port),
            work_root=args.work_root,
            lang=args.lang,
        )
        return 0
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
