"""Shared page-selection parsing helpers for CLI/UI layers."""

from __future__ import annotations

import re
from collections.abc import Sequence


def _tokenize(raw_values: Sequence[str] | str | None) -> list[str]:
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        values: Sequence[str] = (raw_values,)
    else:
        values = raw_values

    tokens: list[str] = []
    for raw in values:
        normalized = (raw or "").strip()
        if not normalized:
            continue
        normalized = normalized.replace("–", "-").replace("—", "-")
        tokens.extend(part for part in re.split(r"[\s,;]+", normalized) if part)
    return tokens


def parse_page_numbers(raw_values: Sequence[str] | str | None) -> tuple[int, ...] | None:
    """Parse 1-based page spec. Supports values like `1,3,5-8`."""
    tokens = _tokenize(raw_values)
    if not tokens:
        return None

    pages: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid page range: {token}")
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid page range: {token}") from exc
            if start < 1 or end < 1:
                raise ValueError(f"Invalid page range: {token}. Page numbers must be >= 1.")
            step = 1 if end >= start else -1
            for page in range(start, end + step, step):
                if page in seen:
                    continue
                seen.add(page)
                pages.append(page)
            continue

        try:
            page = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid page value: {token}") from exc
        if page < 1:
            raise ValueError(f"Invalid page value: {page}. Page numbers must be >= 1.")
        if page in seen:
            continue
        seen.add(page)
        pages.append(page)

    return tuple(pages) if pages else None

