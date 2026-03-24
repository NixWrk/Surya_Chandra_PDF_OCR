"""docTR (Document Text Recognition) engine plugin for OCRmyPDF."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

try:
    import doctr
except ImportError:
    doctr = None

log = logging.getLogger(__name__)


@hookimpl
def add_options(parser):
    """Add docTR-specific options to the argument parser."""
    group = parser.add_argument_group(
        "docTR",
        "Options for docTR OCR engine"
    )
    group.add_argument(
        '--doctr-det-arch',
        default='fast_base',
        help='Detection model architecture (default: fast_base)',
    )
    group.add_argument(
        '--doctr-reco-arch',
        default='Felix92/doctr-torch-parseq-multilingual-v1',
        help='Recognition model: HF Hub ID or built-in arch name '
             '(default: Felix92/doctr-torch-parseq-multilingual-v1)',
    )
    group.add_argument(
        '--doctr-device',
        default='cpu',
        help='Device to run inference on: cpu or cuda (default: cpu)',
    )
    group.add_argument(
        '--doctr-straighten-pages',
        action='store_true',
        help='Enable page straightening before OCR',
    )
    group.add_argument(
        '--doctr-detect-orientation',
        action='store_true',
        help='Enable page orientation detection',
    )


@hookimpl
def check_options(options):
    """Validate docTR options."""
    if doctr is None:
        from ocrmypdf.exceptions import MissingDependencyError
        raise MissingDependencyError(
            "docTR is not installed. "
            "Install it with: pip install python-doctr[torch]"
        )


class DoctrOCREngine(OcrEngine):
    """Implements OCR with docTR."""

    @staticmethod
    def version():
        """Return docTR version."""
        try:
            return doctr.__version__
        except AttributeError:
            return "unknown"

    @staticmethod
    def creator_tag(options):
        """Return the creator tag to identify this software."""
        return f"docTR {DoctrOCREngine.version()}"

    def __str__(self):
        """Return name of OCR engine and version."""
        return f"docTR {DoctrOCREngine.version()}"

    @staticmethod
    def languages(options):
        """Return the set of all languages supported by docTR.

        docTR is largely language-agnostic (character-level recognition),
        but we return common Tesseract codes for OCRmyPDF compatibility.
        """
        return {
            'eng', 'fra', 'deu', 'spa', 'por', 'ita', 'nld', 'pol',
            'ces', 'ron', 'hun', 'fin', 'swe', 'nor', 'dan', 'tur',
            'vie', 'ind', 'msa', 'cat', 'eus', 'glg', 'hrv', 'slk',
            'slv', 'est', 'lav', 'lit',
        }

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        """Get page orientation."""
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        """Get deskew angle."""
        return 0.0

    @staticmethod
    def _get_model(options):
        """Create and configure docTR OCR predictor."""
        import torch
        from doctr.models import ocr_predictor

        device = getattr(options, 'doctr_device', 'cpu')
        straighten = getattr(options, 'doctr_straighten_pages', False)
        detect_orient = getattr(options, 'doctr_detect_orientation', False)
        reco_arch = options.doctr_reco_arch

        # Load recognition model from HF Hub if it looks like a hub ID
        if '/' in reco_arch:
            from doctr.models import from_hub
            log.debug(f"Loading recognition model from HF Hub: {reco_arch}")
            reco_arch = from_hub(reco_arch)

        log.debug(
            f"Creating docTR predictor: det={options.doctr_det_arch}, "
            f"reco={options.doctr_reco_arch}, device={device}"
        )

        model = ocr_predictor(
            det_arch=options.doctr_det_arch,
            reco_arch=reco_arch,
            pretrained=True,
            assume_straight_pages=True,
            straighten_pages=straighten,
            detect_orientation=detect_orient,
        )

        if device != 'cpu' and torch.cuda.is_available():
            model = model.to(torch.device(device))

        return model

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options):
        """Generate hOCR output for an image."""
        log.debug(f"Running docTR on {input_file}")

        # Get image dimensions and DPI
        with Image.open(input_file) as img:
            width, height = img.size
            dpi = img.info.get('dpi', (300, 300))
            # Convert to RGB numpy array
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb, dtype=np.uint8)
            log.debug(f"Input image: {width}x{height}, DPI: {dpi}")

        # Run OCR
        model = DoctrOCREngine._get_model(options)
        result = model([img_array])

        # Build hOCR
        hocr_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"',
            '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
            '<head>',
            '<title></title>',
            '<meta http-equiv="content-type" content="text/html; charset=utf-8" />',
            '<meta name="ocr-system" content="docTR via ocrmypdf-doctr" />',
            '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />',
            '</head>',
            '<body>',
            f'<div class="ocr_page" id="page_1" title="bbox 0 0 {width} {height}">',
        ]

        all_text = []
        word_id = 1
        carea_id = 1
        par_id = 1
        line_id = 1

        page = result.pages[0]

        for block in page.blocks:
            # Block bounding box (normalized -> pixel)
            ((bx_min, by_min), (bx_max, by_max)) = block.geometry
            bx_min_px = int(bx_min * width)
            by_min_px = int(by_min * height)
            bx_max_px = int(bx_max * width)
            by_max_px = int(by_max * height)

            hocr_lines.append(
                f'<div class="ocr_carea" id="carea_{carea_id}" '
                f'title="bbox {bx_min_px} {by_min_px} {bx_max_px} {by_max_px}">'
            )
            hocr_lines.append(
                f'<p class="ocr_par" id="par_{par_id}" '
                f'title="bbox {bx_min_px} {by_min_px} {bx_max_px} {by_max_px}">'
            )

            for line in block.lines:
                # Line bounding box (normalized -> pixel)
                ((lx_min, ly_min), (lx_max, ly_max)) = line.geometry
                lx_min_px = int(lx_min * width)
                ly_min_px = int(ly_min * height)
                lx_max_px = int(lx_max * width)
                ly_max_px = int(ly_max * height)

                hocr_lines.append(
                    f'<span class="ocr_line" id="line_{line_id}" '
                    f'title="bbox {lx_min_px} {ly_min_px} {lx_max_px} {ly_max_px}; '
                    f'baseline 0 0">'
                )

                line_text_parts = []

                # Collect word bboxes in pixel coordinates
                word_boxes = []
                for word in line.words:
                    ((wx_min, wy_min), (wx_max, wy_max)) = word.geometry
                    word_boxes.append({
                        'x_min': wx_min * width,
                        'y_min': wy_min * height,
                        'x_max': wx_max * width,
                        'y_max': wy_max * height,
                    })

                # Ensure minimum gaps between consecutive words so
                # HocrTransform inserts space characters in the PDF.
                # docTR's word bboxes are tight around text and often
                # touch or overlap, leaving no gap for spaces.
                min_gap = 5
                for i in range(len(word_boxes) - 1):
                    curr = word_boxes[i]
                    nxt = word_boxes[i + 1]
                    gap = nxt['x_min'] - curr['x_max']
                    if gap < min_gap:
                        needed = min_gap - gap
                        half = needed / 2
                        curr['x_max'] -= half
                        nxt['x_min'] += half
                        # Ensure words keep a minimum width of 1px
                        if curr['x_max'] <= curr['x_min']:
                            curr['x_max'] = curr['x_min'] + 1
                        if nxt['x_max'] <= nxt['x_min']:
                            nxt['x_min'] = nxt['x_max'] - 1

                for i, word in enumerate(line.words):
                    box = word_boxes[i]
                    wx_min_px = int(box['x_min'])
                    wy_min_px = int(box['y_min'])
                    wx_max_px = int(box['x_max'])
                    wy_max_px = int(box['y_max'])

                    conf_pct = int(word.confidence * 100)

                    # Escape HTML entities
                    word_escaped = (word.value
                                    .replace('&', '&amp;')
                                    .replace('<', '&lt;')
                                    .replace('>', '&gt;'))

                    hocr_lines.append(
                        f'<span class="ocrx_word" id="word_{word_id}" '
                        f'title="bbox {wx_min_px} {wy_min_px} {wx_max_px} {wy_max_px}; '
                        f'x_wconf {conf_pct}">{word_escaped}</span>'
                    )

                    # Add space between words (except after last)
                    if i < len(line.words) - 1:
                        hocr_lines.append(' ')

                    line_text_parts.append(word.value)
                    word_id += 1

                hocr_lines.append('</span>')  # ocr_line
                all_text.append(' '.join(line_text_parts))
                line_id += 1

            hocr_lines.append('</p>')   # ocr_par
            hocr_lines.append('</div>')  # ocr_carea
            carea_id += 1
            par_id += 1

        hocr_lines.extend([
            '</div>',   # ocr_page
            '</body>',
            '</html>',
        ])

        # Write hOCR output
        output_hocr.write_text('\n'.join(hocr_lines), encoding='utf-8')

        # Write text output
        output_text.write_text('\n'.join(all_text), encoding='utf-8')

        log.debug(f"Generated hOCR with {len(all_text)} text lines")

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options):
        """Generate a text-only PDF from an image.

        Uses hOCR as intermediate and converts to PDF via OCRmyPDF's HocrTransform.
        """
        log.debug(f"Generating PDF from {input_file}")

        # Create a temporary hOCR file
        output_hocr = output_pdf.with_suffix('.hocr')

        # Generate hOCR
        DoctrOCREngine.generate_hocr(input_file, output_hocr, output_text, options)

        # Convert hOCR to PDF
        from ocrmypdf.hocrtransform import HocrTransform

        with Image.open(input_file) as img:
            dpi = img.info.get('dpi', (300, 300))[0]

        hocr_transform = HocrTransform(
            hocr_filename=output_hocr,
            dpi=dpi,
        )
        hocr_transform.to_pdf(
            out_filename=output_pdf,
            image_filename=input_file,
            invisible_text=True,
        )


@hookimpl
def get_ocr_engine():
    """Register docTR as an OCR engine."""
    return DoctrOCREngine()
