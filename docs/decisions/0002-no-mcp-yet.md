# ADR 0002: Do not add an MCP adapter yet

Status: accepted.

## Context

The repository already exposes stable HTTP operations for health, job submit,
queue/status, metadata, result download, and queued cancellation. `AGENTS.md`,
the operator runbooks, and the repo-local `uniscan-ocr-operator` skill describe
safe diagnostics and deployment. No identified consumer currently requires MCP.

Running-job cancellation, complete result seals/runtime identity,
authentication, resource limits, and removal of public absolute paths remain
open or incremental contracts. An MCP implemented now would either duplicate
OCR logic or expose unstable behavior.

## Decision

Omit MCP. Agents and integrations should use the existing HTTP API and
repository-local operational guidance.

Reassess only when a concrete consumer demonstrates that typed MCP tools provide
measurable value beyond the HTTP API. Any future adapter must be a thin HTTP
client limited to health, submit, status/wait, result fetch, and supported
cancellation. It must not import OCR internals, browse work roots, edit
manifests, execute arbitrary paths, or control GPU processes.

## Consequences

- There is no additional protocol, dependency, deployment surface, or security
  boundary to maintain now.
- HTTP contract tests remain the source of integration truth.
- A future MCP proposal must name its consumer, authentication boundary,
  compatibility tests, and removal/rollback procedure.
