# ocrmypdf-doctr

An [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) plugin that uses [docTR](https://github.com/mindee/doctr) (Document Text Recognition by Mindee) as the OCR backend.

docTR is a deep learning-based OCR engine that provides text detection and recognition using PyTorch. This plugin integrates it into OCRmyPDF's pipeline, producing searchable PDF/A output.

## Building

Requires [Nix](https://nixos.org/download/) with flakes enabled.

```bash
nix build . --extra-experimental-features 'nix-command flakes'
```

This builds everything from source, including python-doctr v1.0.1 from GitHub. Pretrained model weights are downloaded on first run.

## Usage

```bash
# Basic usage (downloads models on first run)
./result/bin/ocrmypdf --force-ocr input.pdf output.pdf

# Specify language (accepted for OCRmyPDF compatibility)
./result/bin/ocrmypdf --force-ocr -l deu input.pdf output.pdf

# With GPU acceleration
./result/bin/ocrmypdf --force-ocr --doctr-device cuda input.pdf output.pdf
```

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--doctr-det-arch` | `fast_base` | Detection model architecture |
| `--doctr-reco-arch` | `Felix92/doctr-torch-parseq-multilingual-v1` | Recognition model (HF Hub ID or built-in arch) |
| `--doctr-device` | `cpu` | Inference device (`cpu` or `cuda`) |
| `--doctr-straighten-pages` | off | Enable page straightening before OCR |
| `--doctr-detect-orientation` | off | Enable page orientation detection |

### Models

**Detection** (built-in): `fast_tiny`, `fast_small`, `fast_base`, `db_resnet50`, `db_resnet34`, `db_mobilenet_v3_large`, `linknet_resnet18`, `linknet_resnet34`, `linknet_resnet50`

**Recognition** (default): The [multilingual PARSeq model](https://huggingface.co/Felix92/doctr-torch-parseq-multilingual-v1) by Felix Dittrich, loaded from Hugging Face Hub. Supports English, French, German, Italian, Spanish, Portuguese, Czech, Polish, Dutch, Norwegian, Danish, Finnish, and Swedish.

You can override with a built-in arch name (e.g. `--doctr-reco-arch crnn_vgg16_bn`) or any other HF Hub model ID (e.g. `--doctr-reco-arch Noxilus/doctr-torch-parseq-german`).

## Development

```bash
nix develop . --extra-experimental-features 'nix-command flakes'

# Then use ocrmypdf with the plugin
ocrmypdf --plugin ocrmypdf_doctr --force-ocr input.pdf output.pdf

# Run tests
pytest
```

## Project Structure

```
src/ocrmypdf_doctr/
  __init__.py     # Package init, version management
  plugin.py       # OCRmyPDF plugin hooks + DoctrOCREngine
pyproject.toml    # Python package config, entry point registration
flake.nix         # Nix build: python-doctr from GitHub, plugin, wrapped binary
test_pdfs/        # Test documents
```

## How It Works

1. OCRmyPDF splits the PDF into page images
2. The plugin loads docTR's detection + recognition models
3. Detection model locates text regions in the image
4. Recognition model identifies text in each region
5. docTR returns a hierarchical structure: `Document > Page > Block > Line > Word`
6. Each element has normalized bounding box coordinates (0.0-1.0)
7. The plugin converts these to pixel coordinates, ensures minimum gaps between adjacent word bounding boxes (so `HocrTransform` inserts space characters), and generates hOCR
8. OCRmyPDF's `HocrTransform` creates the invisible text layer PDF

## Limitations

### No deskew or orientation detection passthrough

The plugin returns neutral values for page orientation and deskew. Use `--doctr-detect-orientation` and `--doctr-straighten-pages` to let docTR handle these internally.

### CPU-only by default

The Nix build uses CPU-only PyTorch. For GPU support, pass `--doctr-device cuda` (requires a CUDA-capable PyTorch installation).

### Model download on first run

Pretrained weights are downloaded on first invocation and cached in `~/.cache/doctr/models/` (detection, ~65 MB) and `~/.cache/huggingface/` (recognition from HF Hub).

### Built-in recognition models are French-only

docTR's built-in recognition models (e.g. `crnn_vgg16_bn`) are trained on the French character set only and lack characters like German `ä`, `ö`, `ß`. This plugin defaults to the community multilingual PARSeq model instead, which covers 13 European languages. If you override `--doctr-reco-arch` with a built-in arch name, be aware of this limitation.

## License

MPL-2.0
