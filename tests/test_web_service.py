from __future__ import annotations

from urllib.parse import urlparse

from uniscan.web.service import _parse_job_request, _query_bool


def test_query_bool_parsing() -> None:
    assert _query_bool("1", default=False) is True
    assert _query_bool("true", default=False) is True
    assert _query_bool("yes", default=False) is True
    assert _query_bool("0", default=True) is False
    assert _query_bool("false", default=True) is False
    assert _query_bool("no", default=True) is False
    assert _query_bool("unknown", default=True) is True


def test_parse_job_request_defaults() -> None:
    parsed = urlparse("/api/jobs")
    mode, pages_raw, lang, strict, filename, delete_original_text_layer = _parse_job_request(
        parsed,
        default_lang="rus+eng",
    )
    assert mode == "chandra+surya"
    assert pages_raw == ""
    assert lang == "rus+eng"
    assert strict is True
    assert filename == "document.pdf"
    assert delete_original_text_layer is True


def test_parse_job_request_applies_filename_extension() -> None:
    parsed = urlparse(
        "/api/jobs?mode=surya&pages=1-3&lang=eng&strict=0&filename=my_file&delete_text_layer=0"
    )
    mode, pages_raw, lang, strict, filename, delete_original_text_layer = _parse_job_request(
        parsed,
        default_lang="rus+eng",
    )
    assert mode == "surya"
    assert pages_raw == "1-3"
    assert lang == "eng"
    assert strict is False
    assert filename == "my_file.pdf"
    assert delete_original_text_layer is False


def test_parse_job_request_accepts_legacy_delete_text_layer_name() -> None:
    parsed = urlparse("/api/jobs?delete_original_text_layer=false")
    *_, delete_original_text_layer = _parse_job_request(parsed, default_lang="rus+eng")
    assert delete_original_text_layer is False
