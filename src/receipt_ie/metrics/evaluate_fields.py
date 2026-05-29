import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from rapidfuzz.distance import Levenshtein

def compute_em(pred: str, gold: str) -> float:
    """
    Tính Exact Match (EM). Khớp hoàn toàn = 1.0, ngược lại = 0.0.
    """
    return 1.0 if pred.strip() == gold.strip() else 0.0

def compute_nes(pred: str, gold: str) -> float:
    """
    Tính Normalized Edit Similarity (NES):
    NES = 1 - Levenshtein_Distance(pred, gold) / max(len(pred), len(gold))
    """
    pred = pred.strip()
    gold = gold.strip()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    dist = Levenshtein.distance(pred, gold)
    max_len = max(len(pred), len(gold))
    return 1.0 - (dist / max_len)

def compute_cer(pred: str, gold: str) -> float:
    """
    Tính Character Error Rate (CER):
    CER = Levenshtein_Distance(pred, gold) / len(gold)
    """
    pred = pred.strip()
    gold = gold.strip()
    if not gold:
        return 0.0 if not pred else 1.0
    dist = Levenshtein.distance(pred, gold)
    return dist / len(gold)

def evaluate_predictions(predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Đánh giá danh sách dự đoán so với ground truth cho 4 trường thông tin.
    Tính toán EM, NES, CER cho từng trường và Macro Average.
    Sử dụng Map theo ID nếu các bản ghi chứa ID, ngược lại zip trực tiếp.
    Lọc bỏ các mẫu có status == "error".
    """
    fields = ["store_name", "date", "total", "address"]
    
    # Khởi tạo lưu trữ metrics
    metrics = {f: {"em": [], "nes": [], "cer": []} for f in fields}
    
    # Kiểm tra xem các bản ghi có chứa trường 'id' hay không
    has_ids = any("id" in p for p in predictions) and any("id" in g for g in ground_truths)
    
    n_evaluated = 0
    n_skipped_error = 0
    missing_prediction_ids = []
    
    if has_ids:
        valid_pred_map = {}
        for pred in predictions:
            if pred.get("status") == "error":
                n_skipped_error += 1
                continue
            pred_id = pred.get("id")
            if pred_id:
                valid_pred_map[pred_id] = pred

        for gold in ground_truths:
            gold_id = gold.get("id")
            if not gold_id:
                continue

            pred = valid_pred_map.get(gold_id)
            if pred is None:
                missing_prediction_ids.append(gold_id)
                pred_data = {}
            else:
                pred_data = pred.get("normalized_prediction") or pred
            
            # Hỗ trợ cả unified schema lồng nhau và phẳng
            gold_data = gold.get("target") or gold
            
            for f in fields:
                pred_val = pred_data.get(f, "")
                if pred_val is None:
                    pred_val = ""
                gold_val = gold_data.get(f, "")
                if gold_val is None:
                    gold_val = ""
                    
                pred_val = str(pred_val)
                gold_val = str(gold_val)
                
                metrics[f]["em"].append(compute_em(pred_val, gold_val))
                metrics[f]["nes"].append(compute_nes(pred_val, gold_val))
                metrics[f]["cer"].append(compute_cer(pred_val, gold_val))
                
            n_evaluated += 1
    else:
        # Fallback về zip
        for pred, gold in zip(predictions, ground_truths):
            if pred.get("status") == "error":
                n_skipped_error += 1
                continue
                
            pred_data = pred.get("normalized_prediction") or pred
            gold_data = gold.get("target") or gold
            
            for f in fields:
                pred_val = pred_data.get(f, "")
                if pred_val is None:
                    pred_val = ""
                gold_val = gold_data.get(f, "")
                if gold_val is None:
                    gold_val = ""
                    
                pred_val = str(pred_val)
                gold_val = str(gold_val)
                
                metrics[f]["em"].append(compute_em(pred_val, gold_val))
                metrics[f]["nes"].append(compute_nes(pred_val, gold_val))
                metrics[f]["cer"].append(compute_cer(pred_val, gold_val))
                
            n_evaluated += 1
            
    # Tính trung bình cho từng trường và Macro Average
    results = {}
    macro_em = []
    macro_nes = []
    macro_cer = []
    
    for f in fields:
        f_em = sum(metrics[f]["em"]) / len(metrics[f]["em"]) if metrics[f]["em"] else 0.0
        f_nes = sum(metrics[f]["nes"]) / len(metrics[f]["nes"]) if metrics[f]["nes"] else 0.0
        f_cer = sum(metrics[f]["cer"]) / len(metrics[f]["cer"]) if metrics[f]["cer"] else 0.0
        
        results[f] = {
            "EM": round(f_em, 4),
            "NES": round(f_nes, 4),
            "CER": round(f_cer, 4)
        }
        
        macro_em.append(f_em)
        macro_nes.append(f_nes)
        macro_cer.append(f_cer)
        
    results["macro"] = {
        "EM": round(sum(macro_em) / len(macro_em), 4),
        "NES": round(sum(macro_nes) / len(macro_nes), 4),
        "CER": round(sum(macro_cer) / len(macro_cer), 4)
    }
    
    # Bổ sung metadata
    results["n_evaluated"] = n_evaluated
    results["n_skipped_error"] = n_skipped_error
    results["missing_prediction_ids"] = missing_prediction_ids
    
    return results


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(data: Dict[str, Any], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate receipt IE predictions against a gold JSONL file.")
    parser.add_argument("--gold", required=True, help="Gold JSONL path, usually data/processed/test.jsonl")
    parser.add_argument("--pred", required=True, help="Prediction JSONL path")
    parser.add_argument("--output", required=True, help="Output metrics JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    ground_truths = read_jsonl(args.gold)
    predictions = read_jsonl(args.pred)
    results = evaluate_predictions(predictions, ground_truths)
    write_json(results, args.output)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
