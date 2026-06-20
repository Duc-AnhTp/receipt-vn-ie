# Experiment Protocol — Receipt VN IE

## Mục tiêu thí nghiệm

So sánh công bằng 3 phương pháp trích xuất thông tin biên lai tiếng Việt:
1. **Baseline 0**: OCR (PaddleOCR + VietOCR) + Heuristic/Regex
2. **Donut**: End-to-end OCR-free (Swin Encoder → BART Decoder)
3. **VietOCR + LayoutXLM**: OCR-based layout-aware token classification

## Điều kiện công bằng

- **Cùng test set**: Tất cả được chấm trên cùng 234 ID test. Donut dùng toàn bộ train JSONL; LayoutXLM chỉ dùng mẫu `json_and_boxes`.
- **Cùng split**: Main Split 70/15/15 với cùng seed=42
- **Cùng evaluator**: `evaluate_fields.py` tính EM, NES, CER trên `normalized_prediction` vs `target`
- **Cùng hardware**: Đo latency trên cùng 1 máy, cùng GPU/CPU mode

## Metrics chính

| Metric | Hướng tốt | Mô tả |
|---|---|---|
| EM (Exact Match) | ↑ | Tỷ lệ khớp chính xác hoàn toàn |
| NES (Normalized Edit Similarity) | ↑ | `1 - Levenshtein / max(len(pred), len(gold))` |
| CER (Character Error Rate) | ↓ | Tỷ lệ lỗi ký tự |
| E2E Latency | ↓ | Thời gian từ ảnh đầu vào → JSON đầu ra |

Ngoài kết quả trên toàn bộ mẫu, evaluator xuất `present_only` để chỉ chấm các mẫu có ground-truth khác rỗng của từng trường. Prediction có `status="error"` hoặc thiếu ID được tính như prediction rỗng, không bị loại khỏi mẫu số.

## Quy trình thí nghiệm

### Baseline 0
1. Build OCR cache cho toàn bộ tập test
2. Chạy `rule_extractor.py` trên OCR cache
3. Ghi predictions → `outputs/predictions/baseline_test.jsonl`

### Donut
1. Fine-tune trên dữ liệu tiếng Việt với task token `<s_receipt_ie>`
2. Inference trên test → `outputs/predictions/donut_test.jsonl`
3. CORD v2 warm-up chỉ là nhánh optional/future work, không thuộc kết quả chính hiện tại.

### VietOCR + LayoutXLM
1. Build OCR cache (dùng chung với Baseline)
2. Build BIO labels từ OCR cache + ground-truth boxes
3. Fine-tune LayoutXLM
4. Inference: OCR cache → LayoutXLM → aggregate BIO spans → JSON
5. Ghi predictions → `outputs/predictions/layoutxlm_test.jsonl`

### Thí nghiệm phụ: Oracle OCR
1. Build dataset dùng ground-truth OCR text + bounding box thay cho PaddleOCR + VietOCR
2. Train và evaluate LayoutXLM trên oracle data
3. So sánh gap với pipeline thực tế để đo ảnh hưởng lỗi OCR

## Prediction Output Schema

```json
{
  "id": "mcocr_000001",
  "method": "donut",
  "prediction": { "store_name": "...", "date": "...", "total": "...", "address": "..." },
  "normalized_prediction": { "store_name": "...", "date": "...", "total": "...", "address": "..." },
  "raw_output": "<s_store_name>...",
  "latency_ocr_ms": 0.0,
  "latency_model_ms": 120.5,
  "latency_postprocess_ms": 2.0,
  "latency_e2e_ms": 750.5,
  "status": "ok",
  "error": null
}
```

## Bảng kết quả báo cáo chính

| Phương pháp | Store EM | Date EM | Total EM | Address EM | Macro EM | Macro NES | CER ↓ | E2E Latency |
|---|---|---|---|---|---|---|---|---|
| Baseline 0 | | | | | | | | |
| Donut | | | | | | | | |
| VietOCR + LayoutXLM | | | | | | | | |

Lưu ý: latency hiện có của Baseline/LayoutXLM bắt đầu từ OCR cache, trong khi Donut bắt đầu từ ảnh; không gọi đây là so sánh E2E đồng nhất nếu chưa đo lại OCR.

## Bảng phân tích Upper-Bound (Oracle OCR)

| Phương pháp | Macro EM | Macro NES | Ý nghĩa |
|---|---|---|---|
| LayoutXLM + Oracle OCR | | | Cận trên khi OCR hoàn hảo |
| VietOCR + LayoutXLM | | | Pipeline thực tế |
| **Gap** | | | Hao hụt do lỗi OCR |

## Error Taxonomy

| Mã | Loại lỗi | Mô tả |
|---|---|---|
| E1 | OCR Miss | OCR bỏ sót text |
| E2 | OCR Wrong | Nhận dạng sai dấu/ký tự tiếng Việt |
| E3 | Wrong Total | Nhầm subtotal và total |
| E4 | Wrong Date | Lấy nhầm ngày giờ in hóa đơn |
| E5 | Multi-line Address | Trích xuất thiếu dòng địa chỉ |
| E6 | Format Error | Donut sinh chuỗi lỗi thẻ tag |
| E7 | Hallucination | Donut tự sinh thông tin không có trong ảnh |
| E8 | Layout BIO broken | LayoutXLM gán BIO rời rạc không ghép được span |
| E9 | Wrong Reading Order | Thuật toán sắp xếp dòng bị sai |
| E10 | Normalization Error | Trích xuất đúng nhưng normalize sai |
| E11 | Missing Ground Truth | GT thiếu nhưng mô hình trích xuất đúng |

## Reproducibility

- **Seed**: 42
- **Dataset version**: v1.0
- **Split version**: split_70_15_15_group_v1
- **OCR cache version**: ocr_paddle27_vietocr_transformer_v1
- **Framework**: PyTorch + HuggingFace Transformers
