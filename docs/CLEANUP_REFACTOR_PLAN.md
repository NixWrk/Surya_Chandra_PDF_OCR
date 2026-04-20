# Cleanup And Refactor Plan

Last updated: 2026-04-20

## Rollback checkpoint

Current stable checkpoint is tagged:

`checkpoint/dual-venv-stable-20260420` -> commit `158698e`

Rollback commands:

```powershell
git switch main
git reset --hard checkpoint/dual-venv-stable-20260420
```

## Goal

Produce a clean OCR-only repository with predictable behavior for:

1. `surya`
2. `chandra`
3. `chandra+surya` (Chandra text + Surya geometry)

## Constraints

1. No silent fallback that degrades geometry quality.
2. Dual-venv support remains first-class.
3. All user-facing run paths (GUI/CLI) remain operational.
4. Cleanup must be incremental and reversible by commit.

## Work phases

### Phase 0: Safety and baseline

1. Create rollback checkpoint tag.
2. Record cleanup/refactor roadmap inside repo docs.
3. Harden `.gitignore` for local runtime artifacts.

### Phase 1: Repository inventory and deletion map

1. Build explicit keep/remove matrix for files and directories.
2. Mark legacy `uniscan` leftovers that are not needed for OCR pipeline.
3. Confirm runtime-critical paths before deletion.

Deliverable:

`docs/REPO_INVENTORY_KEEP_REMOVE.md`

### Phase 2: Physical cleanup

1. Remove files/directories from "remove" list.
2. Remove dead imports and broken references after deletion.
3. Keep tests aligned with remaining OCR scope.

### Phase 3: Refactor runtime core

1. Separate orchestration from engine execution adapters.
2. Centralize env and cache resolution logic.
3. Keep strict Surya geometry sidecar guarantees.

### Phase 4: Diagnostics and artifact hygiene

1. Normalize progress and error text.
2. Ensure every run folder has deterministic structure.
3. Remove ambiguous or misleading debug output.

### Phase 5: Documentation rewrite

1. Rewrite README as short operator guide.
2. Add architecture and troubleshooting docs.
3. Add "first run", "dual-venv", and "GPU checks" runbooks.

### Phase 6: Verification and release prep

1. Smoke tests for all three modes.
2. Command matrix validation (GUI + CLI).
3. Final changelog and release notes.

## Planned commit series

1. `chore(repo): add cleanup/refactor roadmap and rollback checkpoint docs`
2. `chore(repo): expand gitignore for dual-venv and local runtime artifacts`
3. `docs(repo): add keep/remove inventory for OCR-only target`
4. `chore(cleanup): remove non-OCR legacy modules and stale scripts`
5. `refactor(core): isolate engine routing and runtime config`
6. `refactor(pipeline): simplify benchmark/build orchestration`
7. `refactor(logging): normalize errors and progress reporting`
8. `docs(readme): rewrite README for operator-first quickstart`
9. `docs(runbook): add GPU/cache/troubleshooting guides`
10. `chore(release): final smoke checks and cleanup summary`

## Definition of done

1. Repository tree contains only OCR-pipeline essentials.
2. Three OCR modes pass smoke runs.
3. Geometry quality protections are enforced by code and tests.
4. README is sufficient for setup, run, and troubleshooting.
