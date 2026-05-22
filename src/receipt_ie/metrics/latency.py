import numpy as np
from typing import List, Dict, Any

def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Tính toán các chỉ số thống kê về độ trễ (latency) tính bằng mili-giây.
    Bao gồm: Trung bình (Mean), Trung vị (Median), Độ lệch chuẩn (Std),
    và các phân vị P90, P95, P99.
    """
    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "std_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0
        }
        
    arr = np.array(latencies_ms)
    
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2)
    }
