from typing import Dict, List


def aggregate_bio_spans(words: List[str], labels: List[str]) -> Dict[str, str]:
    fields = {"store_name": [], "date": [], "total": [], "address": []}
    current_field = None
    current_tokens: List[str] = []

    def flush():
        nonlocal current_field, current_tokens
        if current_field and current_tokens:
            fields[current_field].append(" ".join(current_tokens))
        current_field = None
        current_tokens = []

    for word, label in zip(words, labels):
        if label == "O" or "-" not in label:
            flush()
            continue

        prefix, field_label = label.split("-", 1)
        field = field_label.lower()
        if field not in fields:
            flush()
            continue

        if prefix == "B" or current_field != field:
            flush()
            current_field = field
            current_tokens = [word]
        elif prefix == "I" and current_field == field:
            current_tokens.append(word)
        else:
            flush()

    flush()
    return {field: max(spans, key=len) if spans else "" for field, spans in fields.items()}
