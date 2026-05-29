import torch
# Đảm bảo import torch đầu tiên trên Windows để tránh DLL collision
import os
import sys
import shutil
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

# Load metric seqeval thông qua thư viện evaluate
try:
    metric = evaluate.load("seqeval")
except Exception:
    # Fallback nếu không có mạng để tải online, tự load local hoặc import trực tiếp từ seqeval
    metric = None

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
        
    if metric is not None:
        try:
            results = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        except Exception as e:
            print(f"Lỗi tính toán seqeval metric: {e}")
            
    # Fallback thủ công nếu seqeval lỗi hoặc không khả dụng
    # Tính accuracy đơn giản ở cấp độ token
    flat_preds = [p for sublist in true_predictions for p in sublist]
    flat_labels = [l for sublist in true_labels for l in sublist]
    if not flat_labels:
        return {"accuracy": 0.0, "f1": 0.0}
    correct = sum(1 for p, l in zip(flat_preds, flat_labels) if p == l)
    acc = correct / len(flat_labels)
    return {"accuracy": acc, "f1": acc} # Trả về acc làm F1 tượng trưng khi offline

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
    args = parser.parse_args()
    
    layout_cfg = load_yaml(args.layout_config)
    data_cfg = load_yaml(args.data_config)
    
    model_name = layout_cfg["model"]["name"]
    max_length = layout_cfg["model"]["max_length"]
    overlap_threshold = layout_cfg.get("labeling", {}).get("overlap_threshold", 0.5)
    
    # 1. Setup Model và Tokenizer
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
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t_cfg["epochs"],
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
        metric_for_best_model="loss",
        greater_is_better=False,
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
        final_model_dir = os.path.join(base_output_dir, "final")
        if os.path.exists(final_model_dir):
            shutil.rmtree(final_model_dir)
        shutil.copytree(best_model_dir, final_model_dir)
        print(f"Synced final LayoutXLM checkpoint to: {final_model_dir}")

if __name__ == "__main__":
    main()
