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

def evaluate_predictions(predictions: List[Dict[str, str]], ground_truths: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Đánh giá danh sách dự đoán so với ground truth cho 4 trường thông tin.
    Tính toán EM, NES, CER cho từng trường và Macro Average.
    """
    fields = ["store_name", "date", "total", "address"]
    
    # Khởi tạo lưu trữ metrics
    metrics = {f: {"em": [], "nes": [], "cer": []} for f in fields}
    
    for pred, gold in zip(predictions, ground_truths):
        for f in fields:
            pred_val = pred.get(f, "") or ""
            gold_val = gold.get(f, "") or ""
            
            metrics[f]["em"].append(compute_em(pred_val, gold_val))
            metrics[f]["nes"].append(compute_nes(pred_val, gold_val))
            metrics[f]["cer"].append(compute_cer(pred_val, gold_val))
            
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
    
    return results
