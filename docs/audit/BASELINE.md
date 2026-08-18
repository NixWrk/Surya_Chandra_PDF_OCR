# Repository and Runtime Baseline

Audit code baseline: `bbebe4bbb58c0e3a384558e24f22bc06663093c0`.

The later commit `0cff0470e59923de1394fbded6a2d5d67bb8dfad` adds only
`docs/audit/OBSERVED_OCR_FAILURES.md`; the tracked production source is unchanged
from the audit code baseline.

This document separates verified repository/runtime facts from OCR quality claims.
It does not treat a green unit suite as proof of OCR accuracy.

## Git state recorded for the audit

Evidence commands:

```powershell
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git diff --name-status bbebe4b..0cff047
git ls-files
```

At `bbebe4b` the repository contained 60 tracked files. The only change through
`0cff047` is the new observed-failure document. Audit documentation created after
that point is intentionally excluded from the original 60-file baseline.

## Physical inventory

Physical line counts include blank lines and comments.

| Scope | Files | Physical lines |
|---|---:|---:|
| `src/**/*.py` | 23 | 20,708 |
| `tests/**/*.py` | 12 | 16,377 |
| Operational scripts and launchers | 10 | 2,374 |
| Source + tests + operations | 45 | 39,459 |
| All 60 tracked files | 60 | 41,459 |

Evidence command family:

```powershell
git ls-files
Get-Content -LiteralPath <tracked-file> | Measure-Object -Line
```

The earlier approximate figure in `docs/FULL_REPOSITORY_AUDIT_PLAN.md` is not
the authoritative physical-line baseline.

## Verification baseline

| Check | Verified result |
|---|---|
| Full pytest suite | 617 passed, 7 skipped, 5 warnings |
| Pytest reported duration | 248.78 seconds |
| External wall time | 250.966 seconds |
| Branch-coverage run | Same test suite completed in 261.91 seconds |
| Total branch coverage | 74% |
| Ruff | Clean |
| mypy | Clean; 24 files checked |

Recorded command forms:

```powershell
python -m pytest -q
python -m pytest --cov=uniscan --cov-branch --cov-report=term
python -m ruff check .
python -m mypy src scripts/compare_ocr_results.py
```

The exact interpreter/environment used for an accepted future benchmark must be
recorded with package, model, CUDA and device identities. These commands document
the verification surfaces; they are not a dependency lock.

## Container source parity

All 23 tracked Python source files in the inspected container matched the checkout
by SHA-256 content. `MismatchCount=0`.

Container labels may be stale, but there is no evidence of container/source
content drift. Do not diagnose failures as a source mismatch without new hash
evidence.

Evidence procedure:

```powershell
git ls-files "src/*.py" "src/**/*.py"
# SHA-256 each tracked source file in the checkout and container, then compare.
```

## What this baseline proves

- The complete current automated suite was exercised, not a partial subset.
- Static checks are clean at the recorded code baseline.
- The inspected container carries the same tracked Python source content.
- The repository has substantial automated coverage around evidence validation,
  chunking, retry behavior and durable jobs.

## What this baseline does not prove

- OCR text accuracy, layout fidelity or searchable-text usefulness.
- Correct behavior for the two recorded production failures.
- Reproducible dependency/model resolution on a new PC.
- Safe restart recovery for corrupt, tampered or symlinked results.
- Safe publication by a worker after watchdog reclamation.
- Docker image reproducibility; labels alone are not content evidence.

## Preserved evidence

`docs/audit/OBSERVED_OCR_FAILURES.md` records two distinct production failures.
The ignored run artifacts named there must be preserved until minimal regression
fixtures and reproducing tests have been derived. No audit cleanup may delete or
rewrite source PDFs, job data, model caches or retained run evidence.

## Assumptions

- `bbebe4b` is the accepted production-code audit baseline.
- The 617/7 test result, coverage result, Ruff result, mypy result and container
  hash comparison are authoritative measurements supplied to this audit.
- Skips and warnings require later classification but do not make the test run
  partial.
- Accuracy and performance work remains measurement-first.
