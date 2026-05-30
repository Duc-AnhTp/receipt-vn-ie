from typing import Any, Dict, List, Optional


def aggregate_bio_spans(words: List[str], labels: List[str]) -> Dict[str, str]:
    entities = aggregate_bio_entities(words, labels)
    fields = {"store_name": [], "date": [], "total": [], "address": []}
    for entity in entities:
        fields[entity["field"]].append(entity["text"])
    return {field: max(spans, key=len) if spans else "" for field, spans in fields.items()}


def aggregate_bio_entities(
    words: List[str],
    labels: List[str],
    confidences: Optional[List[float]] = None,
    boxes: Optional[List[List[int]]] = None,
) -> List[Dict[str, Any]]:
    fields = {"store_name", "date", "total", "address"}
    entities: List[Dict[str, Any]] = []
    current_field = None
    current_tokens: List[str] = []
    current_confidences: List[float] = []
    current_boxes: List[List[int]] = []
    current_warning = None

    def flush():
        nonlocal current_field, current_tokens, current_confidences, current_boxes, current_warning
        if current_field and current_tokens:
            entity: Dict[str, Any] = {
                "field": current_field,
                "text": " ".join(current_tokens),
                "confidence": sum(current_confidences) / len(current_confidences) if current_confidences else 0.0,
            }
            if current_boxes:
                xs0 = [box[0] for box in current_boxes]
                ys0 = [box[1] for box in current_boxes]
                xs1 = [box[2] for box in current_boxes]
                ys1 = [box[3] for box in current_boxes]
                entity["bbox"] = [min(xs0), min(ys0), max(xs1), max(ys1)]
            if current_warning:
                entity["warning"] = current_warning
            entities.append(entity)
        current_field = None
        current_tokens = []
        current_confidences = []
        current_boxes = []
        current_warning = None

    for idx, (word, label) in enumerate(zip(words, labels)):
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
            current_confidences = [confidences[idx] if confidences and idx < len(confidences) else 0.0]
            current_boxes = [boxes[idx]] if boxes and idx < len(boxes) else []
            if prefix == "I":
                current_warning = "I-without-active-B"
        elif prefix == "I" and current_field == field:
            current_tokens.append(word)
            current_confidences.append(confidences[idx] if confidences and idx < len(confidences) else 0.0)
            if boxes and idx < len(boxes):
                current_boxes.append(boxes[idx])
        else:
            flush()

    flush()
    return entities


def best_fields_from_entities(entities: List[Dict[str, Any]]) -> Dict[str, str]:
    fields = {"store_name": [], "date": [], "total": [], "address": []}
    for entity in entities:
        field = entity.get("field")
        if field in fields:
            fields[field].append(entity)
    return {
        field: max(items, key=lambda item: (item.get("confidence", 0.0), len(item.get("text", ""))))["text"]
        if items else ""
        for field, items in fields.items()
    }
