# Upstream engine version review

Reviewed: 2026-08-18.

This review records upstream state only. It does not authorize a dependency or
model upgrade.

## Installed and current upstream versions

| Engine | Audited runtime | Current upstream | Decision |
| --- | --- | --- | --- |
| Surya | `surya-ocr==0.17.1` | `0.22.1` | Keep 0.17.1 until a migration benchmark exists |
| Chandra | `chandra-ocr==0.2.0` | `0.2.0` | No package upgrade available |

Sources:

- Surya package metadata:
  <https://pypi.org/pypi/surya-ocr/json>
- Surya current project metadata:
  <https://raw.githubusercontent.com/datalab-to/surya/master/pyproject.toml>
- Chandra package metadata:
  <https://pypi.org/pypi/chandra-ocr/json>
- Chandra current project metadata:
  <https://raw.githubusercontent.com/datalab-to/chandra/master/pyproject.toml>

## Surya migration risk

The current Surya upstream is not a drop-in patch update for this repository.
Its published upgrade notes describe a new inference manager, a shared VLM
backend, vLLM/llama.cpp lifecycle, and changed output schemas (`text_lines` to
`blocks`, HTML content, and layout/table field changes). Current upstream also
requires `transformers>=5.12.1`, while the audited Surya runtime intentionally
uses Transformers 4.57.1 and supplies line geometry to the existing hybrid
adapter.

Therefore upgrading Surya would require, at minimum:

1. a new adapter/output-schema characterization suite;
2. model/backend deployment and cache changes;
3. geometry, ordering, punctuation, and exact-retention comparisons;
4. cold/warm latency and VRAM measurements;
5. a rollback-compatible runtime profile;
6. a full versioned OCR corpus including the two observed incidents.

No such accepted baseline exists yet. Upgrading now would violate the approved
measurement-first plan and could invalidate both geometry and deployment
contracts.

## Chandra status

The audited package already matches upstream `0.2.0`. The locally accepted
model snapshot still needs a complete cryptographic model lock; current cache
evidence records a specific Hugging Face revision and metadata/size checks, but
large weight SHA-256 values have not yet been captured.

## Promotion gate

An engine update may be proposed only as one isolated experiment. It must retain
the prior environment for rollback and beat the accepted baseline without any
page-level omission, evidence, searchable-text, layout, or VRAM regression.
