import torch
# Đảm bảo import torch đầu tiên trên Windows để tránh DLL collision
import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import evaluate

from receipt_ie.models.layoutxlm_model import setup_layoutxlm_model_and_tokenizer
from receipt_ie.models.layoutxlm_dataset import LayoutXLMDataset, layoutxlm_collate_fn
from receipt_ie.data.schemas import ID2LABEL

# Load metric seqeval thông qua thư viện evaluate.
try:
    metric = evaluate.load("seqeval")
except Exception:
    metric = None

ALLOW_METRIC_FALLBACK = False
_METRIC_FALLBACK_WARNED = False


def _fallback_token_metrics(true_predictions, true_labels):
    global _METRIC_FALLBACK_WARNED
    if not ALLOW_METRIC_FALLBACK:
        raise RuntimeError(
            "seqeval metric is required for official LayoutXLM training. "
            "Install evaluate/seqeval or run with --allow_metric_fallback "
            "only for smoke tests."
        )
    if not _METRIC_FALLBACK_WARNED:
        print(
            "WARNING: seqeval unavailable. Using token accuracy only for a "
            "smoke test. This F1 is not valid for reporting."
        )
        _METRIC_FALLBACK_WARNED = True
    flat_preds = [p for sublist in true_predictions for p in sublist]
    flat_labels = [l for sublist in true_labels for l in sublist]
    if not flat_labels:
        return {"precision": 0.0, "recall": 0.0, "accuracy": 0.0, "f1": 0.0}
    correct = sum(1 for p, l in zip(flat_preds, flat_labels) if p == l)
    acc = correct / len(flat_labels)
    return {"precision": acc, "recall": acc, "accuracy": acc, "f1": acc}


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        pred_list = []
        label_list = []
        for p_val, l_val in zip(prediction, label):
            if l_val != -100:
                pred_list.append(ID2LABEL.get(p_val, "O"))
                label_list.append(ID2LABEL.get(l_val, "O"))
        true_predictions.append(pred_list)
        true_labels.append(label_list)

    if metric is None:
        return _fallback_token_metrics(true_predictions, true_labels)

    try:
        results = metric.compute(predictions=true_predictions, references=true_labels)
    except Exception as exc:
        if not ALLOW_METRIC_FALLBACK:
            raise RuntimeError("Failed to compute seqeval metric.") from exc
        return _fallback_token_metrics(true_predictions, true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình LayoutXLM")
    parser.add_argument("--layout_config", type=str, default="configs/layoutxlm.yaml", help="Cấu hình LayoutXLM")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml", help="Cấu hình Dữ liệu")
    parser.add_argument("--project_root", type=str, default=".", help="Root thư mục dự án")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["ocr_cache", "oracle_ocr"], 
        default="ocr_cache", 
        help="Chế độ chạy: ocr_cache (thực tế) hoặc oracle_ocr (phân tích cận trên)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epochs từ config file")
    parser.add_argument("--max_steps", type=int, default=None, help="Ghi đè số steps huấn luyện tối đa")
<<<<<<< HEAD
    parser.add_argument(
        "--allow_metric_fallback",
        action="store_true",
        help="Chỉ dùng cho smoke test: fallback token accuracy nếu seqeval không khả dụng.",
    )
=======
    parser.add_argument("--train_batch_size", type=int, default=None, help="Ghi đè batch size huấn luyện per device")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Ghi đè batch size đánh giá per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Ghi đè số bước tích lũy gradient")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None, help="Bật gradient checkpointing để tiết kiệm VRAM")
>>>>>>> ec28b02a0beeba1ef96bcb97c0583d324ca87f46
    args = parser.parse_args()

    global ALLOW_METRIC_FALLBACK
    ALLOW_METRIC_FALLBACK = args.allow_metric_fallback
    
    layout_cfg = load_yaml(args.layout_config)
    data_cfg = load_yaml(args.data_config)
    
    if args.epochs is not None:
        layout_cfg["training"]["epochs"] = args.epochs
    if args.max_steps is not None:
        layout_cfg["training"]["max_steps"] = args.max_steps
    if args.train_batch_size is not None:
        layout_cfg["training"]["train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        layout_cfg["training"]["eval_batch_size"] = args.eval_batch_size
    if args.gradient_accumulation_steps is not None:
        layout_cfg["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.gradient_checkpointing is not None:
        layout_cfg["training"]["gradient_checkpointing"] = args.gradient_checkpointing
    
    model_name = layout_cfg["model"]["name"]
    max_length = layout_cfg["model"]["max_length"]
    overlap_threshold = layout_cfg.get("labeling", {}).get("overlap_threshold", 0.5)
    print(f"\n=== KHỞI TẠO LAYOUTXLM (mode: {args.mode}) ===")
    model, tokenizer = setup_layoutxlm_model_and_tokenizer(model_name=model_name)
    
    # 2. Tạo Dataset
    train_jsonl = data_cfg["processed"]["train_jsonl"]
    val_jsonl = data_cfg["processed"]["val_jsonl"]
    
    train_dataset = LayoutXLMDataset(
        jsonl_path=train_jsonl,
        tokenizer=tokenizer,
        mode=args.mode,
        max_length=max_length,
        project_root=args.project_root,
        annotation_level_filter="json_and_boxes",
        is_train=True,
        overlap_threshold=overlap_threshold
    )
    
    val_dataset = LayoutXLMDataset(
        jsonl_path=val_jsonl,
        tokenizer=tokenizer,
        mode=args.mode,
        max_length=max_length,
        project_root=args.project_root,
        annotation_level_filter="json_and_boxes",
        is_train=False,
        overlap_threshold=overlap_threshold
    )
    
    # Thư mục checkpoints đầu ra
    base_output_dir = layout_cfg["training"]["output_dir"]
    output_dir = os.path.join(base_output_dir, args.mode)
    
    # 3. Cấu hình Training Arguments
    t_cfg = layout_cfg["training"]
    use_fp16 = t_cfg["fp16"] and torch.cuda.is_available()
    
    epochs = t_cfg["epochs"]
    max_steps = t_cfg.get("max_steps", -1)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=t_cfg["train_batch_size"],
        per_device_eval_batch_size=t_cfg["eval_batch_size"],
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", False),
        learning_rate=float(t_cfg["learning_rate"]),
        weight_decay=t_cfg["weight_decay"],
        warmup_ratio=t_cfg["warmup_ratio"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        fp16=use_fp16,
        logging_steps=t_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "runs"),
        remove_unused_columns=False, # Quan trọng: Giữ lại các cột đầu vào của LayoutLMv2
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none"
    )
    
    # 4. Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=layoutxlm_collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t_cfg["early_stopping_patience"])]
    )
    
    # 5. Huấn luyện
    print(f"Bắt đầu huấn luyện...")
    trainer.train()
    
    # 6. Lưu mô hình và tokenizer tốt nhất (chỉ tiến hành ở luồng chính để tránh xung đột ghi trong DDP)
    best_model_dir = os.path.join(output_dir, "best_model")
    if trainer.is_world_process_zero():
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        print(f"Đã lưu mô hình tốt nhất vào: {best_model_dir}")

if __name__ == "__main__":
    main()
