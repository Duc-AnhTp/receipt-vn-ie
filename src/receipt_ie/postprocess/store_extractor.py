import re
from typing import Iterable

from receipt_ie.data.normalize_text import normalize_store_name
from receipt_ie.postprocess.total_extractor import normalize_for_match

SKIP_KEYWORDS = [
    "hoa don",
    "bill",
    "invoice",
    "ban hang",
    "mst",
    "ma so thue",
    "dt",
    "sdt",
    "tel",
    "hotline",
    "dia chi",
    "address",
]


def _score_store_line(line: str, index: int) -> float:
    normalized = normalize_for_match(line)
    if any(keyword in normalized for keyword in SKIP_KEYWORDS):
        return -1
    clean = re.sub(r"[_\-*=+\|/\\]", "", line).strip()
    if len(clean) < 3:
        return -1
    digit_ratio = sum(ch.isdigit() for ch in line) / max(len(line), 1)
    if digit_ratio > 0.35:
        return -1
    uppercase_ratio = sum(ch.isupper() for ch in line) / max(sum(ch.isalpha() for ch in line), 1)
    return 10 - index + uppercase_ratio * 3 + min(len(clean), 30) / 30


def extract_store_name_from_lines(lines: Iterable[str], max_lines: int = 5) -> str:
    candidates = []
    for idx, line in enumerate(list(lines)[:max_lines]):
        text = re.sub(r"\s+", " ", str(line or "")).strip()
        if not text:
            continue
        score = _score_store_line(text, idx)
        if score >= 0:
            candidates.append((score, text))
    if not candidates:
        return ""
    return normalize_store_name(max(candidates, key=lambda item: item[0])[1])
