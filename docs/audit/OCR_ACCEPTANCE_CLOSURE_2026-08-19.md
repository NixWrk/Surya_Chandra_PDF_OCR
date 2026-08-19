# OCR Acceptance Closure — 2026-08-19

## Decision

By explicit user decision, the current production OCR scope is accepted as
complete. There is no active OCR repair, tuning, performance, deployment, or
measurement task after this checkpoint. Deferred ideas and unavailable external
evidence are not release blockers.

This closure applies to the repository's supported production contract:

- `chandra+surya` searchable-PDF processing;
- the durable local HTTP job API and documented GUI/CLI surfaces;
- the current Docker production deployment on configured GPU0;
- same-deployment resumable chunk caches;
- the current engine/model versions and OCR policy.

## Accepted production checkpoint

| Item | Accepted value |
|---|---|
| Storage/runtime commit | `5d7b7d2` |
| Production image | `sha256:b774e4aa955df82b24b360027e8b084576ad5f6b18a5251a6f9bf7cc848fd42b` |
| Production tags | `surya-chandra-ocr:latest`, `surya-chandra-ocr:prod-771b5de` |
| Rollback image | `surya-chandra-ocr:rollback-f470cf-20260819` |
| Full suite | 680 passed, 9 skipped, 5 warnings in 216.47 s |
| Static checks | Ruff clean; mypy clean in 24 source files |
| Deployment checks | Compose config pass; Docker new-PC preflight `ok: true` |
| Production service | healthy; 21 durable jobs; no active job at closure verification |

The production image itself was not rebuilt for the storage change because no
image filesystem, Python source, engine, model, or OCR policy changed. The
container was recreated with `--no-build --pull never`, so no dependency or
model download was required.

## Accepted correctness and performance evidence

The audit added deterministic protection for recovery seals, unsafe links and
paths, stale watchdog publication, malformed/encrypted uploads, shutdown,
punctuation-only geometry, verified blank pages, immutable per-run settings, and
resume identity. All accepted fixes were test-first and keep strict fail-closed
publication behavior.

The long-document path has both repeat and targeted evidence:

- three fresh Windows-bind runs: 596.799 s median, 628.505 s observed maximum,
  every quality gate passed;
- the duplicate immediate-premerge evidence pass was removed without removing
  any integrity fence;
- cache-hit validation/merge: 198.584 s on the Windows bind versus 31.031 s on
  the Docker-managed volume;
- fresh production job `4eb0ea9d0611`: 433.897 s, 23 pages, 2,954,784 bytes,
  exact Ground Truth retention, and zero partial failures;
- successful-run cleanup restored the native cache to its pre-run 1,854 files
  and 737,854,373 bytes.

The storage change does not skip required validation. It avoids the excessive
Windows-bind metadata latency while retaining evidence, fingerprint, manifest,
snapshot, containment, stable-read, TOCTOU, and runtime-drift checks.

## Data and storage contract

Durable inputs, `jobs.sqlite3`, metadata/events, retained failures, benchmark
evidence, and published PDFs remain ordinary host files below `./outputs`. Only
`/data/work/runs/hybrid_chunk_cache` uses the Compose-owned native volume
`surya-chandra-ocr-hybrid-chunk-cache`.

The migration retained every existing job and matched the old and new caches at
1,854 files, 737,854,373 bytes, and aggregate manifest SHA-256
`bfbe1e51b4960448919f590bf81bde68f29a12436323929d62f7493ddc4a955c`.
The original host cache remains a rollback fallback. No global document catalog
or permanent processed-document history was added.

## Accepted non-blocking limitations

- The 433.897-second native-volume result is one accepted after-run, not a p95 or
  a new median/tail distribution.
- Reviewed representative private raster Ground Truth is unavailable. Existing
  procedural and retained incident fixtures remain the accepted regression set.
- Per user decision, the offline no-cache Docker source-layer build is accepted
  as partial replacement for a clean dependency/model build. It does not prove
  first-time provisioning on every new PC.
- The existing Windows Surya venv was not mutated during the audit; current
  production acceptance is the healthy Docker path.
- Queue/page thresholds, cooperative running cancellation, and persistent-cache
  quotas are future product/operations decisions, not current OCR defects.
- Thin MCP remains intentionally deferred until a concrete consumer needs more
  than the documented HTTP API.

## Reopening rule

OCR work reopens only when at least one of these occurs:

1. a new reproducible production incident;
2. a benchmark regression or new reviewed Ground Truth;
3. a requested engine, model, OCR-policy, or supported-platform change;
4. a new deployment, queue-limit, cancellation, retention, security, or MCP
   requirement.

Any reopened change must use the preserved benchmark and incident evidence,
remain test-first, and report quality, page integrity, runtime, and resource
deltas. Closure does not authorize weakening validation or deleting user data,
retained incidents, rollback images, model caches, or benchmark evidence.

## Evidence map

- `AUDIT_REPORT.md` — consolidated audit findings and disposition.
- `QUALITY_PERFORMANCE_BASELINE.md` — accepted quality/performance checkpoints.
- `PRODUCTION_PROMOTION_2026-08-19.md` — image attestation and production series.
- `RESIDUAL_STORAGE_PROFILE_2026-08-19.md` — residual attribution, A/B, migration,
  production after-run, and rollback.
- `IMPLEMENTATION_BACKLOG.md` — historical ranked matrix and deferred candidates.
- `EXECUTION_STATUS_2026-08-18.md` — chronological implementation evidence.
