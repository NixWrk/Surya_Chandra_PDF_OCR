# Residual Storage Profile And Production Migration — 2026-08-19

## Scope

This checkpoint explains the 256.788-second median `long-23p` residual left
after subtracting recorded Surya, Chandra, and PDF-build time from the three
fresh production-bind runs. It does not change OCR engines, models, render DPI,
textless-source policy, retry policy, reconciliation, or any validation fence.

The source fixture SHA-256 is
`9e2f178711ae9aeb9e1a8b434386128c044c8851788b4559e6ad8c03663082e4`.
All profiling used image
`sha256:b774e4aa955df82b24b360027e8b084576ad5f6b18a5251a6f9bf7cc848fd42b`,
`--pull never`, `--network none`, and pre-existing read-only model caches where
OCR was permitted.

## Fresh profiled run

One diagnostic fresh-cache run on the exact Windows `/data/work` bind recorded:

| Signal | Seconds |
|---|---:|
| Wall | 651.236 |
| Surya result sum | 63.932 |
| Chandra result sum | 299.188 |
| PDF-build result sum | 4.389 |
| Residual | 283.727 |
| `_required_chunk_evidence` inclusive | 124.733 |
| `_complete_chunk_record` inclusive | 108.566 |
| `_complete_chunk_evidence_entries` inclusive | 88.262 |

The profiler adds overhead, so 651.236 seconds is diagnostic rather than a new
operational baseline. The output still had 23 pages, 2,954,784 bytes, zero
partial failures, and exact Ground Truth text retention.

Self-time showed the underlying cause:

| Operation | Calls | Self seconds |
|---|---:|---:|
| `posix.lstat` | path-component checks | 125.82 |
| `posix.stat` | file/directory state checks | 78.24 |
| SHA-256 primitive | content hashing | 2.01 |

The large residual was therefore not mainly hashing. Strict ownership,
symlink/reparse, hardlink, containment, stable-read, manifest, and TOCTOU checks
perform many metadata calls; Docker Desktop makes each call expensive when the
tree is a Windows bind.

## Cache-hit storage A/B

The same completed 23-page cache was copied into isolated test roots. Both runs:

- used the same run key
  `8136577788a371bfd07949cd2bb23292ee6373cee35149dc128b73ca4501f9a7`;
- ran the unmodified production validation and merge code;
- set a fail-closed guard that raised before any OCR engine invocation;
- produced 23 pages, 2,954,784 bytes, and exact Ground Truth retention;
- retained every evidence, fingerprint, manifest, snapshot, and runtime-drift
  fence.

| Signal | Windows bind | Docker volume | Delta |
|---|---:|---:|---:|
| Validation/merge wall | 198.584 s | 31.031 s | -167.553 s (-84.374%) |
| `lstat` calls | 71,376 | 71,376 | unchanged |
| `lstat` self-time | 89.12 s | 0.10 s | -89.02 s |
| `stat` calls | 35,498 | 35,498 | unchanged |
| `stat` self-time | 59.06 s | 0.08 s | -58.98 s |
| SHA-256 self-time | 1.45 s | 1.42 s | -0.03 s |

The call counts and validation code were unchanged. Storage latency alone
explains the difference.

## Accepted change

Commit `5d7b7d2` changes only the Compose storage topology:

```text
./outputs                           -> /data/work
hybrid-chunk-cache named volume    -> /data/work/runs/hybrid_chunk_cache
```

Durable job inputs, `jobs.sqlite3`, metadata/events, retained failures, and
published results remain ordinary host files under `./outputs`. Only the
same-deployment resumable working cache is native to Docker. Successful HTTP
chunk working data remains subject to the existing cleanup policy.

Targeted RED/GREEN evidence:

```text
RED:   1 failed, 4 deselected
GREEN: 15 passed in 0.28 s
Ruff:  clean
docker compose config --quiet: pass
```

No image rebuild or download was required because the image filesystem and OCR
code did not change.

## Production migration

The queue was checked twice and had zero active jobs. The API was stopped before
copying. The pre-existing host cache and the new named volume matched exactly:

```text
top-level run directories: 7
files:                     1,854
bytes:                     737,854,373
aggregate manifest SHA:    bfbe1e51b4960448919f590bf81bde68f29a12436323929d62f7493ddc4a955c
```

The production container was recreated with `--no-build --pull never`. It is
healthy on image `sha256:b774e4aa955df...`; the nested mount is the Compose-owned
volume `surya-chandra-ocr-hybrid-chunk-cache`. All 20 pre-migration job records
were retained. The original host cache was not deleted and remains a migration
fallback; this intentionally costs about 738 MB until an explicit retention
decision permits removal.

## Production after-run

Fresh async HTTP job `4eb0ea9d0611` used the same `long-23p` source:

| Signal | Result |
|---|---:|
| Wall, start to finish | 433.896604 s |
| Surya event-stage sum | 70.303733 s |
| Chandra event-stage sum | 315.862554 s |
| PDF-build event-stage sum | 1.782101 s |
| Event-stage residual | 45.948216 s |
| Delta vs Windows-bind median | -162.901900 s (-27.295963%) |
| Output | 23 pages, 2,954,784 bytes |
| Output SHA-256 | `f5dc05fe6a0d4387eaefba24f986e6bb9e84a26f8fb07689c5d88110473346e0` |
| Exact Ground Truth retention | pass |
| Partial failures | 0 |

After publication, the cache returned to the same seven directories, 1,854
files, and 737,854,373 bytes, proving that successful-run cleanup still acts on
the nested volume.

## Final verification

```text
python -m pytest -q
680 passed, 9 skipped, 5 warnings in 216.47 s

python -m ruff check src tests
All checks passed

python -m mypy
Success: no issues found in 24 source files

docker compose config --quiet
pass

.\scripts\preflight_new_pc.ps1 -Target Docker -Json
ok: true
```

The local API remained healthy with 21 durable jobs and no active job. The
production container and `latest`/`prod-771b5de` tags still use image
`sha256:b774e4aa955df...`.

## Limitations and rollback

The storage A/B is controlled and cache-hit. The production after-run is one
fresh run, not a new median or tail distribution. Use the prior three-run series
as the Windows-bind baseline until repeated native-volume measurements are
needed.

Rollback does not require deleting the named volume or any job:

1. revert `5d7b7d2`;
2. run `docker compose up -d --no-build --pull never --force-recreate ocr-api`;
3. verify health and the retained job count.

The preserved host cache makes this rollback immediately resumable. Do not
prune the named volume while a running/interrupted job still depends on it.

## Evidence paths

Ignored local profiler evidence is under:

```text
outputs/audit_synthetic_baseline/v1_0_1/prod-771b5de-residual-profile/
```

Accepted subtrees are `fresh-bind-r2`, `cache-hit-bind-r2`, and
`volume-cache-hit-r1`. Earlier `r1` attempts were stopped or failed closed after
GPU contention/path mismatch and are excluded from all comparisons.
