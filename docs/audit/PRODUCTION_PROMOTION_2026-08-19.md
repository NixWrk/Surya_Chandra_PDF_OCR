# Production image promotion and repeated long-document checkpoint

Date: 2026-08-19

Status: accepted local production deployment evidence. This checkpoint does
not claim a clean dependency build or a statistically estimated p95.

## Scope

The checkpoint verifies that the evidence-reuse optimization accepted at
`1cb708c` is present in a rebuilt container, measures three independent
`long-23p` runs with resource sampling, and promotes the verified image to the
local production service without changing Surya, Chandra, model weights, OCR
policy, or user/job retention.

The source tree was clean at:

```text
771b5de5465278495dda80d06a7a2413db3a7ca1
main...origin/main [ahead 49]
```

The only changes after production commit `1cb708c` were audit documents.

## Offline build result

A normal build was attempted first with:

```text
docker build --pull=false --network none --progress=plain \
  --tag surya-chandra-ocr:audit-771b5de .
```

It stopped at the `apt-get` layer because that exact layer was no longer cached.
Network isolation prevented package-index access, as intended; no dependency or
model download was allowed. This failed attempt did not change `latest` or the
running service.

The accepted fallback was a `--no-cache`, `--network none`, `--pull=false`
source/install-layer rebuild on top of the preserved immutable image:

```text
base:       sha256:f470cf1520e43ae67b70bf63e5dded12235ebde07e5139c68307cb867b06bdc0
measured:   sha256:72ad02bb45d162612fcc9ea9bf74feca6c9f4e44d314dce292dfa572019d1d16
production: sha256:b774e4aa955df82b24b360027e8b084576ad5f6b18a5251a6f9bf7cc848fd42b
```

Both `/opt/venvs/surya` and `/opt/venvs/chandra` reinstalled the current
`uniscan` package with `--no-deps --no-build-isolation`. This is accepted as an
offline source-layer rebuild only. It does not prove clean OS, Python, CUDA,
engine, or model provisioning.

## Image attestation

The source-layer candidate and final production labels record full source revision
`771b5de5465278495dda80d06a7a2413db3a7ca1`. The following checks passed:

- all 26 tracked production inputs (`src/` plus `pyproject.toml`, `README.md`,
  and `scripts/docker-entrypoint.sh`) have identical SHA-256 rows on the host
  and inside `/app`;
- `src/uniscan/app/ocr_pipeline.py` has SHA-256
  `8d129583b95f566f6a758bbc20b6c37ae8d0dc7ac394819b27da5304b577ea1e`
  in the host tree, `/app/src`, and both installed venvs;
- the image reports pipeline revision `chandra-surya-resumable-v11` and contains
  `_ValidatedReusableChunk`;
- both engine venvs import `torch 2.11.0+cu126` and `uniscan`;
- both venvs completed a tensor operation on the single exposed `cuda:0`;
- Docker preflight and CLI help passed.

No tracked machine GPU UUID was introduced. The host UUID is stored only in the
ignored local `.env`, so ordinary Compose commands remain reproducible after a
shell or host restart.

## Repeated `long-23p` measurement

All runs used:

- the same source SHA-256
  `9e2f178711ae9aeb9e1a8b434386128c044c8851788b4559e6ad8c03663082e4`;
- Ground Truth SHA-256
  `bc061d2a8c5a26e1ea84f4f630a80c41b63d6599de5b1dcdac448e7edb173e6a`;
- image `sha256:72ad02bb45d...`, `--pull never`, and `--network none`;
- pre-existing read-only model caches and a fresh chunk cache per run;
- three sequential chunks of 10, 10, and 3 pages;
- an exact Windows bind at `/data/work`, matching the production Compose
  storage contract and retaining evidence;
- a resource loop with a one-second sleep between probes. Docker/NVIDIA command
  latency makes the effective cadence slower than exactly 1 Hz.

| Run | Wall s | Surya s | Chandra s | PDF build s | Residual s | Peak RAM bytes | Peak GPU0 VRAM delta MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 596.799 | 63.801 | 275.447 | 3.515 | 254.035 | 4,221,952,852 | 11,046 |
| r2 | 585.563 | 58.077 | 267.214 | 3.483 | 256.788 | 3,333,968,364 | 11,046 |
| r3 | 628.505 | 61.200 | 280.533 | 4.044 | 282.728 | 3,289,944,949 | 11,046 |

Summary:

| Signal | Result |
|---|---:|
| Wall median | 596.799 s |
| Wall observed maximum | 628.505 s |
| Surya median | 61.200 s |
| Chandra median | 275.447 s |
| PDF-build median | 3.515 s |
| Residual median | 256.788 s |
| Residual observed maximum | 282.728 s |
| Observed container RAM peak | 4,221,952,852 bytes |
| GPU0 absolute VRAM peak | 13,943 MiB |
| GPU0 pre-run background | 2,897 MiB |
| GPU0 peak above background | 11,046 MiB |
| Peak GPU utilization | 98% |

The VRAM value is total physical GPU0 usage. The delta is relative to the
immediate pre-run background and cannot separate unrelated GPU processes. The
same absolute/delta peak occurred in all three independent runs. Container RAM
comes from `docker stats`; a final `0B / 0B` sample after each container stopped
was recorded as a sampling warning and excluded from the maximum.

All three evaluator reports passed:

- CER `0` and WER `0`;
- exact searchable-text retention `pass` for 23/23 pages;
- page mapping `pass`;
- zero partial page failures;
- 23 output pages and 2,954,784 output bytes.

The three-run observed maximum is a tail proxy, not a p95 estimate. The earlier
463.709-second after-run used an anonymous nested Docker work volume because the
parent bind did not override the image's `/data/work` volume. It is valid for
the narrow before/after code comparison documented in
`PREMERGE_EVIDENCE_PERFORMANCE_2026-08-19.md`, but it is not directly comparable
to this production-like Windows-bind series. The new median therefore becomes
the operational Windows-bind baseline. The 256.788-second median residual was
subsequently profiled and addressed without removing validation; see
`RESIDUAL_STORAGE_PROFILE_2026-08-19.md`.

## Production promotion

Before recreation, the HTTP queue had no running or queued work. A consistent
SQLite backup was written to ignored local storage. The old image was preserved
as:

```text
surya-chandra-ocr:rollback-f470cf-20260819
sha256:f470cf1520e43ae67b70bf63e5dded12235ebde07e5139c68307cb867b06bdc0
```

The measured image `sha256:72ad02bb45d...` was initially promoted with a
no-build, force-recreate Compose command. The running container matched the same
26-file source manifest and retained all pre-existing durable job records.

A real async HTTP smoke then submitted the one-page generated `clean-en` fixture:

```text
job_id:          d74385dc849a
status:          done
wall:            84.603 s
result bytes:    175,354
result SHA-256:  d8026ab955eb5dd4311581fd170b69d8c3ff030b319c127e35ac39d95d6b4b7b
page count:      1
exact retention: pass
```

After the smoke, a stale inherited custom label was found: it advertised
pipeline `v10` even though the attested runtime code was `v11`. A network-free,
metadata-only image was built directly from the exact measured image. It adds no
filesystem changes and overrides the source/pipeline labels to the attested
revision. The exact measured image remains tagged `measured-771b5de`.

The corrected image `sha256:b774e4aa955df...` is tagged as both
`surya-chandra-ocr:prod-771b5de` and the Compose pointer
`surya-chandra-ocr:latest`. The final running container reports source revision
`771b5de...`, pipeline revision `chandra-surya-resumable-v11`, production role,
and the same 26-file SHA-256 manifest. It is healthy. The job store changed only by the
expected additional completed synthetic smoke job.

## Residual profile and storage promotion

A standard-library profiler on the same `long-23p` fixture found 71,376
`lstat` and 35,498 `stat` calls during one cache-hit validation/merge pass. On
the Windows bind, those calls consumed about 148.18 seconds; SHA-256 consumed
about 1.45 seconds. The same sealed cache and unchanged validation code took:

| Storage | Cache-hit validation/merge |
|---|---:|
| Windows bind | 198.584 s |
| Docker-managed volume | 31.031 s |

Commit `5d7b7d2` adds a nested named volume only at
`/data/work/runs/hybrid_chunk_cache`. Durable job inputs, SQLite metadata,
retained evidence, and published PDFs remain under the host `./outputs` bind.
The existing cache was copied with the service stopped and matched before start:
1,854 files, 737,854,373 bytes, aggregate manifest SHA-256
`bfbe1e51b4960448919f590bf81bde68f29a12436323929d62f7493ddc4a955c`.

Production HTTP job `4eb0ea9d0611` then completed the fresh 23-page fixture in
433.897 seconds, 162.902 seconds (27.296%) below the prior Windows-bind median.
It produced 23 pages and 2,954,784 bytes with exact Ground Truth retention and
zero partial failures. Successful-run cache cleanup restored the volume to the
same 1,854 files and 737,854,373 bytes that existed before the smoke.

## Local evidence

Raw ignored evidence is under:

```text
outputs/audit_synthetic_baseline/v1_0_1/prod-771b5de-multirun/
```

It contains the runtime-generated fixture, three independent output/work trees,
raw resource samples, summaries, evaluator inputs/reports, container logs,
production HTTP smoke result/metadata, host/container source manifests, and the
final corrected-image label/manifest summary.

## Remaining limits and rollback

Still open:

1. a clean dependency/model build when downloads or a prepared clean cache are
   available;
2. representative raster-scan Ground Truth and controlled `mixed-layout`
   accuracy experiments;
3. repeated native-volume runs if a median/tail claim is required; the current
   433.897-second production result is one accepted after-run.

Rollback is local and does not delete newer job evidence:

```text
docker tag surya-chandra-ocr:rollback-f470cf-20260819 surya-chandra-ocr:latest
docker compose up -d --no-build --force-recreate ocr-api
```
