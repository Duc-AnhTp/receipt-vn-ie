import re
from typing import Iterable

from receipt_ie.data.normalize_text import normalize_address, normalize_date
from receipt_ie.postprocess.total_extractor import normalize_for_match

MAX_CONTINUATION_LINES = 3

ADDRESS_KEYWORDS = [
    "địa chỉ",
    "dia chi",
    "address",
    "đ/c",
    "dc",
    "đường",
    "duong",
    "phường",
    "phuong",
    "quận",
    "quan",
    "huyện",
    "huyen",
    "tỉnh",
    "tinh",
    "tp",
    "thành phố",
    "thanh pho",
]

BLOCK_KEYWORDS = [
    "tel",
    "phone",
    "dt",
    "sdt",
    "mst",
    "tax",
    "total",
    "tong cong",
    "thanh toan",
]


def _has_address_keyword(text: str) -> bool:
    text_n = normalize_for_match(text)
    return any(normalize_for_match(keyword) in text_n for keyword in ADDRESS_KEYWORDS)


def _blocked(text: str) -> bool:
    text_n = normalize_for_match(text)
    if normalize_date(text):
        return True
    return any(keyword in text_n for keyword in BLOCK_KEYWORDS)


def extract_address_from_lines(lines: Iterable[str]) -> str:
    line_list = [
        re.sub(r"\s+", " ", str(line or "")).strip()
        for line in lines
        if str(line or "").strip()
    ]
    for idx, line in enumerate(line_list):
        if not _has_address_keyword(line) or _blocked(line):
            continue
        parts = [line]
        for next_line in line_list[idx + 1: idx + 1 + MAX_CONTINUATION_LINES]:
            if _blocked(next_line):
                break
            next_norm = normalize_for_match(next_line)
            if len(next_line) <= 5 or next_norm.startswith(("ngay", "date")):
                break
            parts.append(next_line)
        return normalize_address(" ".join(parts))

    for line in line_list:
        if _has_address_keyword(line) and not _blocked(line):
            return normalize_address(line)
    return ""
