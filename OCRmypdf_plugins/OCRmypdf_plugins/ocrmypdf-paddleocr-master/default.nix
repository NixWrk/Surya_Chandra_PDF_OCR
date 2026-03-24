{ pkgs ? import <nixpkgs-unstable> {} }:

let
  # Use Python 3.12 (paddlepaddle 3.0.0 in unstable supports 3.12)
  python = pkgs.python312;
  pythonPackages = python.pkgs;

  # Override paddlex to include OCR optional dependencies
  paddlex-with-ocr = pythonPackages.paddlex.overridePythonAttrs (old: {
    dependencies = (old.dependencies or []) ++ (old.optional-dependencies.ocr or []);
  });

  # Override paddleocr to use our paddlex-with-ocr
  paddleocr-with-ocr = pythonPackages.paddleocr.override {
    paddlex = paddlex-with-ocr;
  };
in
pythonPackages.buildPythonPackage rec {
  pname = "ocrmypdf-paddleocr";
  version = "0.1.0";
  format = "pyproject";

  src = ./.;

  nativeBuildInputs = with pythonPackages; [
    setuptools
    setuptools-scm
    wheel
  ];

  propagatedBuildInputs = with pythonPackages; [
    (ocrmypdf.override {
      # Skip tests for img2pdf which is failing in unstable
      img2pdf = img2pdf.overridePythonAttrs (old: { doCheck = false; });
    })
    paddlepaddle
    paddleocr-with-ocr  # PaddleOCR with PaddleX[ocr]
    pillow
  ];

  nativeCheckInputs = with pythonPackages; [
    pytest
  ];

  # PaddlePaddle tests require GPU or specific CPU instructions
  doCheck = false;

  pythonImportsCheck = [
    "ocrmypdf_paddleocr"
  ];

  meta = with pkgs.lib; {
    description = "PaddleOCR plugin for OCRmyPDF";
    homepage = "https://github.com/yourusername/ocrmypdf-paddleocr";
    license = licenses.mpl20;
    maintainers = with maintainers; [ ];
    platforms = platforms.unix;
  };
}
