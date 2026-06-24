import re
import unicodedata
from typing import Iterable

from receipt_ie.data.normalize_text import normalize_date, normalize_money

TOTAL_KEYWORDS = [
    "tổng cộng",
    "tong cong",
    "tổng thanh toán",
    "tong thanh toan",
    "thanh toán",
    "thanh toan",
    "phải trả",
    "phai tra",
    "total",
    "amount",
    "grand total",
    "thành tiền",
    "thanh tien",
]

BLOCK_KEYWORDS = [
    "tel",
    "phone",
    "dt",
    "sdt",
    "đt",
    "mst",
    "tax",
    "mã số thuế",
    "ma so thue",
    "invoice",
    "bill no",
    "order",
]


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_for_match(text: str) -> str:
    return strip_accents(text).lower()


def is_probably_phone(number: str) -> bool:
    digits = re.sub(r"[^\d]", "", number or "")
    return (
        re.fullmatch(r"0[3-9]\d{8}", digits) is not None
        or re.fullmatch(r"84[3-9]\d{8}", digits) is not None
    )


def is_probably_tax_code(number: str) -> bool:
    return len(number) in {10, 13}


def is_too_small_for_total(value: int) -> bool:
    return value < 1000


def money_candidates(text: str) -> list[str]:
    candidates = []
    for match in re.finditer(r"\d[\d\.,\s]*", text or ""):
        value = normalize_money(match.group(0))
        if value and value.isdigit():
            candidates.append(value)
    return candidates


def _valid_money(value: str, context: str, allow_keyword_context: bool = False) -> bool:
    if not value or not value.isdigit():
        return False
    number = int(value)
    context_n = normalize_for_match(context)
    has_total_keyword = any(keyword in context_n for keyword in [normalize_for_match(k) for k in TOTAL_KEYWORDS])
    if normalize_date(context):
        return False
    if any(keyword in context_n for keyword in [normalize_for_match(k) for k in BLOCK_KEYWORDS]) and not has_total_keyword:
        return False
    if is_probably_phone(value) and not has_total_keyword:
        return False
    if is_probably_tax_code(value) and not has_total_keyword:
        return False
    if is_too_small_for_total(number):
        return False
    return allow_keyword_context or number <= 1_000_000_000


def _line_has_total_keyword(line: str) -> bool:
    line_n = normalize_for_match(line)
    return any(normalize_for_match(keyword) in line_n for keyword in TOTAL_KEYWORDS)


def extract_total_from_lines(lines: Iterable[str]) -> str:
    line_list = [re.sub(r"\s+", " ", str(line or "")).strip() for line in lines if str(line or "").strip()]
    if not line_list:
        return ""

    for idx in range(len(line_list) - 1, -1, -1):
        line = line_list[idx]
        if not _line_has_total_keyword(line):
            continue
        window = [line]
        if idx + 1 < len(line_list):
            window.append(line_list[idx + 1])
        candidates = [
            value
            for text in window
            for value in money_candidates(text)
            if _valid_money(value, text, allow_keyword_context=True)
        ]
        if candidates:
            return max(candidates, key=lambda value: int(value))

    start_idx = int(len(line_list) * 0.66)
    candidates = [
        value
        for text in line_list[start_idx:]
        for value in money_candidates(text)
        if _valid_money(value, text)
    ]
    return max(candidates, key=lambda value: int(value)) if candidates else ""
