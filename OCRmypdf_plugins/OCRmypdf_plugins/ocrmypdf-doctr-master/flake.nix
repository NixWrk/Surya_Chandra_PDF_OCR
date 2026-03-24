{
  description = "docTR plugin for OCRmyPDF";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      python = pkgs.python312;
      pythonPackages = python.pkgs;

      # opencv-python shim: docTR depends on opencv-python but nixpkgs provides opencv4
      python-shims = pkgs.runCommand "doctr-compat-shims" {} ''
        mkdir -p $out/${python.sitePackages}

        cat > $out/${python.sitePackages}/opencv_python.py << 'PYEOF'
# Shim: docTR checks for opencv-python; nixpkgs provides opencv4
from cv2 import *
PYEOF

        mkdir -p $out/${python.sitePackages}/opencv_python-4.10.0.84.dist-info
        cat > $out/${python.sitePackages}/opencv_python-4.10.0.84.dist-info/METADATA << 'EOF'
Metadata-Version: 2.1
Name: opencv-python
Version: 4.10.0.84
EOF
      '';

      # Build python-doctr from GitHub
      python-doctr = pythonPackages.buildPythonPackage rec {
        pname = "python-doctr";
        version = "1.0.1";
        pyproject = true;

        src = pkgs.fetchFromGitHub {
          owner = "mindee";
          repo = "doctr";
          rev = "v${version}";
          hash = "sha256-s0/81C5ZS1eLAKC03XTGBSbbNyDWE/MZxKKW4bsql38=";
        };

        env.SETUPTOOLS_SCM_PRETEND_VERSION = version;

        build-system = with pythonPackages; [
          setuptools
          setuptools-scm
          wheel
        ];

        # Remove opencv-python from wheel metadata — nixpkgs provides opencv4
        nativeBuildInputs = [ pythonPackages.pythonRelaxDepsHook ];
        pythonRemoveDeps = [ "opencv-python" ];

        dependencies = with pythonPackages; [
          torch
          torchvision
          onnx
          numpy
          scipy
          h5py
          opencv4
          pypdfium2
          pyclipper
          shapely
          langdetect
          rapidfuzz
          huggingface-hub
          pillow
          defusedxml
          anyascii
          tqdm
          validators
        ];

        # Skip tests (require network access for model downloads)
        doCheck = false;

        pythonImportsCheck = [ "doctr" ];

        meta = {
          description = "docTR (Document Text Recognition) by Mindee";
          homepage = "https://github.com/mindee/doctr";
          license = pkgs.lib.licenses.asl20;
          platforms = pkgs.lib.platforms.linux;
        };
      };

      # The plugin package
      ocrmypdf-doctr = pythonPackages.buildPythonPackage {
        pname = "ocrmypdf-doctr";
        version = "0.1.0";
        pyproject = true;

        src = builtins.path {
          path = ./.;
          name = "ocrmypdf-doctr-src";
          filter = path: type:
            let baseName = builtins.baseNameOf path;
            in !(
              baseName == "references" ||
              baseName == ".git" ||
              baseName == "result" ||
              baseName == "flake.nix" ||
              baseName == "flake.lock" ||
              baseName == "jail.drv" ||
              baseName == "jail.nix" ||
              baseName == "jail.nix-env" ||
              (type == "regular" && pkgs.lib.hasSuffix ".pdf" baseName)
            );
        };

        env.SETUPTOOLS_SCM_PRETEND_VERSION = "0.1.0";

        build-system = with pythonPackages; [
          setuptools
          setuptools-scm
          wheel
        ];

        dependencies = [
          (pythonPackages.ocrmypdf.override {
            img2pdf = pythonPackages.img2pdf.overridePythonAttrs (old: { doCheck = false; });
          })
          python-doctr
          pythonPackages.pillow
        ];

        doCheck = false;

        pythonImportsCheck = [ "ocrmypdf_doctr" ];

        meta = {
          description = "docTR plugin for OCRmyPDF";
          license = pkgs.lib.licenses.mpl20;
          platforms = pkgs.lib.platforms.linux;
        };
      };

      # Python environment with all runtime dependencies
      pythonEnv = python.withPackages (_ps: [
        ocrmypdf-doctr
      ]);

      shimPath = "${python-shims}/${python.sitePackages}";

      # Wrapped ocrmypdf binary with plugin pre-loaded
      ocrmypdf-wrapped = pkgs.writeShellScriptBin "ocrmypdf" ''
        export PYTHONPATH="${shimPath}''${PYTHONPATH:+:$PYTHONPATH}"
        export PATH="${pkgs.lib.makeBinPath [ pkgs.ghostscript pkgs.pngquant pkgs.unpaper ]}''${PATH:+:$PATH}"
        exec ${pythonEnv}/bin/ocrmypdf --plugin ocrmypdf_doctr "$@"
      '';

    in {
      packages.${system} = {
        default = ocrmypdf-wrapped;
        plugin = ocrmypdf-doctr;
        doctr = python-doctr;
      };

      apps.${system}.default = {
        type = "app";
        program = "${ocrmypdf-wrapped}/bin/ocrmypdf";
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          (python.withPackages (_ps: [
            ocrmypdf-doctr
            pythonPackages.pytest
            pythonPackages.ipython
          ]))
          pkgs.ghostscript
          pkgs.pngquant
          pkgs.unpaper
        ];

        shellHook = ''
          export PYTHONPATH="${shimPath}''${PYTHONPATH:+:$PYTHONPATH}"

          echo "OCRmyPDF-docTR development environment"
          echo "======================================="
          echo ""
          echo "Python: $(python --version)"
          echo "OCRmyPDF: $(python -c 'import ocrmypdf; print(ocrmypdf.__version__)' 2>/dev/null || echo 'not found')"
          echo "docTR: $(python -c 'import doctr; print(doctr.__version__)' 2>/dev/null || echo 'not found')"
          echo ""
          echo "Usage:"
          echo "  ocrmypdf --plugin ocrmypdf_doctr input.pdf output.pdf"
          echo ""
        '';
      };
    };
}
