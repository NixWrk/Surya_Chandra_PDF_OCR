# Full Repository Audit And Improvement Plan

Status: approved

## Goal

Bring the repository into a clean, measurable, reproducible state while:

1. preserving correctness and strict OCR evidence guarantees;
2. improving OCR accuracy where benchmarks prove a gain;
3. reducing end-to-end processing time and resource waste;
4. making deployment on a new Windows PC fast and predictable;
5. making the repository easy and safe for coding agents and LLMs to operate.

"Reduce speed" is interpreted as reducing processing duration (making OCR faster).

## Agent Operating Model

1. GPT-5.6 Sol remains the orchestrator.
2. GPT-5.6 Luna with reasoning effort `max` is the implementation worker.
3. The orchestrator owns scope, sequencing, review, acceptance criteria, and final integration.
4. The worker receives bounded tasks with explicit inputs, outputs, and verification commands.
5. Understand Anything must never be invoked unless the user explicitly requests it.
6. Audit findings come before refactors. No large rewrite is allowed without measured evidence.
7. Changes are delivered as small, reversible commits with focused tests.

## Preliminary Repository Facts

The planning scan found:

1. Approximately 35,177 lines across 43 source, test, and operational files.
2. The largest production modules are:
   - `src/uniscan/app/ocr_pipeline.py`: about 6,127 lines;
   - `src/uniscan/ocr/benchmark.py`: about 5,014 lines;
   - `src/uniscan/ocr/artifact_searchable.py`: about 3,572 lines;
   - `src/uniscan/web/service.py`: about 1,970 lines.
3. CI, dependency lock files, `AGENTS.md`, a repository skill, and an MCP adapter are absent.
4. The declared MIT license has no committed license file.
5. A machine-specific GPU UUID is embedded in setup and Docker configuration.
6. Dependency declarations are split between `pyproject.toml`, the Dockerfile, and Windows setup scripts.
7. Some older documentation describes contracts that no longer match the production-only hybrid path.
8. Running OCR jobs cannot currently be cancelled safely.

These are starting observations, not final audit conclusions.

## Phase 1: Establish A Measurable Baseline

Create a versioned representative corpus containing:

1. Russian, English, and mixed-language pages;
2. low-contrast, noisy, skewed, and low-resolution scans;
3. headings, footnotes, tables, columns, and dense layouts;
4. graphics-only and verified textless pages;
5. rotated pages;
6. PDFs with existing text layers;
7. large documents and known historical failure pages.

Measure:

1. CER and WER where ground truth is available;
2. page-level omissions and hallucinations;
3. exact searchable-text retention and ordering;
4. text-layer geometry quality;
5. failed and recovered page counts;
6. Surya, Chandra, PDF-build, and validation time separately;
7. peak VRAM and RAM;
8. output size;
9. cold-start, warm-start, cache-hit, and cache-miss duration.

Deliverables:

1. `docs/audit/BASELINE.md`;
2. machine-readable benchmark results;
3. a repeatable benchmark command;
4. a baseline commit and tag.

## Phase 2: Full Repository Inventory

Classify every tracked file as:

1. production runtime;
2. test infrastructure;
3. benchmark or diagnostics;
4. deployment;
5. documentation;
6. obsolete;
7. duplicated;
8. generated or local-only.

Record:

1. module and import boundaries;
2. every environment variable and default;
3. CLI, GUI, and HTTP contracts;
4. job and chunk state transitions;
5. filesystem artifacts and ownership;
6. hardware-specific assumptions;
7. magic constants and duplicated policies.

Deliverable: `docs/audit/REPOSITORY_INVENTORY.md`.

## Phase 3: Correctness, Reliability, And Security Audit

Review and test:

1. partial, empty, malformed, and inconsistent OCR results;
2. atomic publication of PDFs, manifests, and job metadata;
3. restart, resume, cache validation, and cache corruption;
4. page/text/geometry provenance;
5. Unicode, BOM, LF/CRLF, and PDF extraction behavior;
6. path traversal, symlink, hardlink, and TOCTOU protections;
7. malformed, oversized, encrypted, and adversarial PDFs;
8. timeouts, OOM, hung subprocesses, cancellation, and shutdown;
9. idempotency, queue ordering, and result retention;
10. overwrite behavior and user-visible result paths.

Every confirmed defect must receive a reproducing test before its fix.

## Phase 4: Architecture And Technical Debt Audit

Evaluate module boundaries for:

1. runtime configuration;
2. engine adapters;
3. page reconciliation;
4. retry policy;
5. evidence validation;
6. chunk cache and resume;
7. searchable-PDF generation;
8. durable job storage and queueing;
9. HTTP transport;
10. GUI and CLI;
11. benchmark and evaluation tooling.

Identify dead code, duplicated Windows/Docker behavior, oversized functions,
global mutable state, temporary environment mutation, and tests coupled to
implementation details. Refactoring must remain incremental rather than a rewrite.

## Phase 5: Accuracy Experiments

Verify current upstream Surya and Chandra releases, changelogs, CUDA requirements,
and output formats. Do not upgrade by version number alone.

Benchmark controlled variants of:

1. Surya and Chandra versions;
2. render DPI;
3. lossless versus JPEG textless sources;
4. contrast, scaling, deskew, and rotation preprocessing;
5. Chandra prompts and retry policies;
6. Surya and Chandra geometry selection;
7. alignment band and placement strategies;
8. fonts and invisible text-layer parameters.

Promote a change only when it improves the target metrics without introducing
page-level regressions, evidence failures, or unacceptable VRAM use.

## Phase 6: Performance Profiling And Optimization

Profile before changing behavior. Investigate:

1. repeated page rendering;
2. reuse of sealed raster artifacts;
3. repeated PDF opening and extraction;
4. the cost of strict validation stages;
5. safe page batching;
6. model precision and loading behavior;
7. chunk size versus VRAM and throughput;
8. warm-up time versus per-job latency;
9. resuming from successful OCR when only PDF assembly fails.

Report median and tail latency, not only one successful run.

## Phase 7: Reproducible Dependencies

Create explicit and independently reproducible Surya and Chandra runtimes.

Add:

1. exact constraints or lock files;
2. a pinned Docker base image;
3. a documented Python/PyTorch/CUDA compatibility matrix;
4. a model-cache manifest and preflight verification;
5. one source of truth for versions used by pyproject, Docker, and Windows setup;
6. an isolated dependency-update benchmark gate.

## Phase 8: New-PC Deployment

Use Docker as the canonical production path and retain dual venvs as a supported
development fallback.

Provide:

1. `.env.example`;
2. automatic GPU discovery instead of a fixed UUID;
3. CUDA wheel selection and validation;
4. automatic network/cache/directory initialization;
5. model download and verification;
6. `bootstrap.ps1`, `preflight.ps1`, and `smoke.ps1`;
7. one command each for install, run, diagnose, update, and rollback;
8. backup/restore guidance for jobs and caches.

Acceptance requires a cold deployment from a clean clone on a different PC.

## Phase 9: CI And Test Pyramid

Add:

1. fast CPU CI for pytest, Ruff, mypy, and packaging;
2. contract tests for APIs, manifests, and state transitions;
3. synthetic PDF tests;
4. a Docker build test;
5. a GPU smoke test on a local or self-hosted runner;
6. a scheduled full benchmark on the reference corpus.

Routine CI must not require downloading or running the production OCR models.

## Phase 10: Agent And LLM Integration

Add an agent pack consisting of:

1. `AGENTS.md` with architecture constraints, commands, generated paths,
   production invariants, and the explicit Understand Anything rule;
2. `docs/ARCHITECTURE.md` with components, data flow, state transitions, and
   validation boundaries;
3. a repository-local `uniscan-ocr-operator` skill for job diagnosis, tests,
   benchmark, rebuild, deployment, and cache recovery;
4. stable machine-oriented commands such as `test`, `lint`, `smoke`, `deploy`,
   and `diagnose-job`;
5. only if justified, a thin MCP adapter over the existing HTTP API.

Potential MCP tools:

1. `ocr_health`;
2. `submit_pdf`;
3. `get_job`;
4. `wait_job`;
5. `fetch_result`;
6. `cancel_job`;
7. `inspect_failure`.

The MCP adapter must not duplicate OCR logic or grant arbitrary filesystem access.
If `AGENTS.md`, the skill, and CLI commands are sufficient, MCP should be omitted.

## Audit Deliverables

Before major implementation, produce:

1. `docs/audit/AUDIT_REPORT.md`;
2. `docs/audit/QUALITY_PERFORMANCE_BASELINE.md`;
3. `docs/audit/TARGET_ARCHITECTURE.md`;
4. `docs/audit/IMPLEMENTATION_BACKLOG.md`.

The backlog must rank each item by severity, impact, confidence, effort, risk,
dependencies, verification method, and rollback strategy.

## Implementation Order

1. data-loss and correctness defects;
2. reproducibility and dependency integrity;
3. the accuracy benchmark and controlled model experiments;
4. measured performance improvements;
5. incremental architecture cleanup;
6. new-PC deployment;
7. agent tooling and optional MCP;
8. final release and migration documentation.

## Definition Of Done

1. A clean PC can deploy and start the service with one documented command.
2. No production configuration contains a machine-specific GPU UUID or path.
3. Dependency and model versions are reproducible.
4. Accuracy and runtime are measured automatically against a versioned baseline.
5. No accepted optimization regresses the quality gates.
6. Running jobs can be stopped or recovered without corrupting durable state.
7. CPU CI and GPU smoke tests pass.
8. The repository tree and documentation describe the actual production contract.
9. A new agent can find architecture, commands, invariants, and troubleshooting
   guidance without reverse-engineering the repository.
10. The final release has migration, rollback, and benchmark reports.
