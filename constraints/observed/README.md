# Observed dependency snapshots

These files record an environment that existed and passed its installed-package
consistency check. They are evidence, not an installation lock.

## Snapshot identity

- Observation date: `2026-08-18`
- Image tag inspected: `surya-chandra-ocr:6c9d75b`
- Image digest: `sha256:f470cf1520e43ae67b70bf63e5dded12235ebde07e5139c68307cb867b06bdc0`
- Image creation time: `2026-08-18T07:25:19.540530956Z`
- Platform: Linux `amd64`
- Python: `3.11.15`
- PyTorch wheel family: `cu126`
- OCI revision label: `f31855e524f35ec1858eb3ec3f776cb7ac477fed`
- OCI `uniscan.code.parent` label: `2ba06887342885de1a548f1b7dc339e5fe817f2c`

The image tag is not proof of the source commit. The immutable image digest and
embedded labels are recorded separately because the tag and revision label do
not identify the same commit.

Both venvs reported `No broken requirements found` from `pip check`.
`uniscan @ file:///app` was excluded because it is the local project install.

## Non-goals

- The Dockerfile does not consume these files.
- `setup_dual_venv.cmd` does not consume these files.
- No clean Docker build has been validated against these constraints.
- No claim is made for Windows or `cu128`.

Before enforcement, create and validate a resolution for every supported
platform/profile. Each requires a clean install/build, `pip check`, imports,
CUDA tensor smoke, model-cache preflight, the full CPU suite, and a fixed OCR
benchmark comparison. An observed snapshot alone is not sufficient.
