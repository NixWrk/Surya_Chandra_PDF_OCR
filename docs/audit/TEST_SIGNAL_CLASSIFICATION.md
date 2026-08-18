# Test signal classification

Evidence date: 2026-08-18. The complete local CPU suite through `c89b593`
reported:

```text
678 passed, 9 skipped, 5 warnings in 259.49s
```
The earlier `676 passed` result remains the verified-blank checkpoint recorded
in the synthetic-baseline document.

## Expected failures

None. The malformed and encrypted PDF admission cases in
`tests/test_http_pdf_resource_limits_audit.py` are ordinary passing tests. They
now require `400` before durable job creation. No `xfail` marker remains under
`tests/`.

## Skips

All nine skips have explicit environmental or fixture causes:

| Test location | Count | Reason | Classification |
|---|---:|---|---|
| `tests/test_ocr_benchmark.py:571` | 1 | file symlink unavailable | Windows privilege/capability gap |
| `tests/test_ocr_benchmark.py:618` | 1 | directory symlink unavailable | Windows privilege/capability gap |
| `tests/test_ocr_benchmark.py:2650` | 1 | optional external/private OCR fixture absent | not part of default CPU suite; separate synthetic/private baselines exist |
| `tests/test_web_service.py:558` | 1 | directory symlink unavailable (`WinError 1314`) | containment test not exercised on this host |
| `tests/test_web_service.py:1847` | 1 | result symlink unavailable (`WinError 1314`) | HTTP containment test not exercised on this host |
| `tests/test_ocr_page_reconciliation.py:1650` | 1 | file symlink unavailable (`WinError 1314`) | retry-evidence containment test not exercised |
| `tests/test_ocr_zero_output_retry.py:102` | 1 | directory symlink unavailable (`WinError 1314`) | retry-copy containment test not exercised |
| `tests/test_app_searchable_pdf.py:1435` | 1 | directory symlink unavailable (`WinError 1314`) | compare-text containment test not exercised |
| `tests/test_app_searchable_pdf.py:1508` | 1 | manifest symlink unavailable (`WinError 1314`) | chunk-manifest containment test not exercised |

These skips do not indicate observed production failures, but eight of them leave
security/integrity branches unexecuted on this Windows host. CI should retain the
tests and run them on a symlink-capable environment. The optional external
fixture is no longer the only real-OCR evidence: accepted private and synthetic
offline baselines are documented separately.

## Warnings

All five warnings are `DeprecationWarning` instances emitted while importing the
SWIG layer used by PyMuPDF/`fitz`:

- `SwigPyPacked` has no `__module__`;
- `SwigPyObject` has no `__module__`;
- `swigvarlink` has no `__module__`.

Pytest attributes them to whichever test first imports the binding; targeted runs
showed the same five warnings under preprocessing and PDF split/merge tests. They
are third-party compatibility noise, not OCR-result failures. Do not suppress
`DeprecationWarning` globally. Re-evaluate them with a separately benchmarked
PyMuPDF upgrade and keep warning counts visible in CI.
