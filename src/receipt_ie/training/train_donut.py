import torch
# Đảm bảo import torch đầu tiên trên Windows để tránh DLL collision
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)

from receipt_ie.models.donut_model import setup_donut_model_and_processor
from receipt_ie.models.donut_dataset import DonutDataset, collate_fn

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_stage(
    model_name: str,
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    task_token: str,
    cord_task_token: str,
    donut_config: Dict[str, Any],
    project_root: str,
    is_warmup: bool = False
):
    """
    Huấn luyện Donut. Luồng chính là fine-tuning tiếng Việt; CORD chỉ là optional/future work.
    """
    print(f"\n=== BẮT ĐẦU HUẤN LUYỆN: {'WARM-UP CORD' if is_warmup else 'FINE-TUNING TIẾNG VIỆT'} ===")
    print(f"Model name/path: {model_name}")
    print(f"Train data: {train_jsonl}")
    print(f"Val data: {val_jsonl}")
    print(f"Output directory: {output_dir}")
    print(f"Task token: {task_token}")
    
    # 1. Setup Model và Processor
    image_size = donut_config["model"].get("image_size", None)
    model, processor = setup_donut_model_and_processor(
        model_name=model_name,
        task_token=task_token,
        cord_task_token=cord_task_token,
        image_size=image_size
    )
    
    # Cập nhật decoder_start_token_id cho giai đoạn huấn luyện hiện tại
    task_token_id = processor.tokenizer.convert_tokens_to_ids(task_token)
    model.config.decoder_start_token_id = task_token_id
    
    # 2. Tạo Dataset
    max_length = donut_config["model"]["max_length"]
    train_dataset = DonutDataset(
        jsonl_path=train_jsonl,
        processor=processor,
        task_token=task_token,
        max_length=max_length,
        project_root=project_root,
        is_train=True,
        strict_image=True
    )
    
    val_dataset = None
    if val_jsonl:
        val_dataset = DonutDataset(
            jsonl_path=val_jsonl,
            processor=processor,
            task_token=task_token,
            max_length=max_length,
            project_root=project_root,
            is_train=False,
            strict_image=True
        )
        
    # 3. Cấu hình Training Arguments
    t_cfg = donut_config["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = t_cfg["fp16"] and torch.cuda.is_available()
    
    # Giảm epoch nếu là warm-up pretrain để tránh overfitting vào cấu trúc CORD
    epochs = t_cfg["warmup_epochs"] if is_warmup else t_cfg["epochs"]
    max_steps = t_cfg.get("max_steps", -1)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=t_cfg["train_batch_size"],
        per_device_eval_batch_size=t_cfg["eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=float(t_cfg["learning_rate"]),
        weight_decay=t_cfg["weight_decay"],
        warmup_ratio=t_cfg["warmup_ratio"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        fp16=use_fp16,
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=t_cfg["eval_steps"] if val_dataset else None,
        save_strategy="steps" if val_dataset else "epoch",
        save_steps=t_cfg["save_steps"] if val_dataset else None,
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="loss" if val_dataset else None,
        greater_is_better=False if val_dataset else None,
        logging_dir=os.path.join(output_dir, "runs"),
        remove_unused_columns=False, # Quan trọng: để tránh filter mất các cột custom trong dataset
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none" # Tắt report wandb mặc định để tránh lỗi credential
    )
    
    # 4. Khởi tạo Trainer
    callbacks = []
    if val_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=t_cfg["early_stopping_patience"]))
        
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=callbacks
    )
    
    # 5. Huấn luyện
    resume_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Phát hiện checkpoint cũ tại: {resume_checkpoint}. Tiếp tục huấn luyện nối tiếp từ đây...")
            
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # 6. Lưu mô hình và processor tốt nhất (chỉ tiến hành ở luồng chính để tránh xung đột ghi trong DDP)
    best_model_dir = os.path.join(output_dir, "best_model")
    if trainer.is_world_process_zero():
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.save_model(best_model_dir)
        processor.save_pretrained(best_model_dir)
        print(f"Đã lưu mô hình tốt nhất vào: {best_model_dir}")
    
    return best_model_dir

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Donut")
    parser.add_argument("--donut_config", type=str, default="configs/donut.yaml", help="Cấu hình Donut")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml", help="Cấu hình Dữ liệu")
    parser.add_argument("--project_root", type=str, default=".", help="Root thư mục dự án")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["warmup", "finetune", "full"], 
        default="finetune", 
        help="Chế độ chạy: finetune là main run; warmup/full dùng CORD v2 optional/future work"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epochs từ config file")
    parser.add_argument("--max_steps", type=int, default=None, help="Ghi đè số steps huấn luyện tối đa")
    args = parser.parse_args()
    
    donut_cfg = load_yaml(args.donut_config)
    data_cfg = load_yaml(args.data_config)
    
    if args.epochs is not None:
        donut_cfg["training"]["epochs"] = args.epochs
        donut_cfg["training"]["warmup_epochs"] = args.epochs
    if args.max_steps is not None:
        donut_cfg["training"]["max_steps"] = args.max_steps
        
    model_name = donut_cfg["model"]["name"]
    task_token = donut_cfg["model"]["task_token"]
    cord_task_token = donut_cfg["model"]["cord_task_token"]
    
    # Thư mục checkpoints
    output_dir = donut_cfg["training"]["output_dir"]
    
    # 1. Optional/future work: Warm-up pretraining trên CORD v2 (nếu chọn warmup hoặc full)
    current_model = model_name
    if args.mode in ["warmup", "full"]:
        cord_train = data_cfg["processed"]["donut_warmup_train"]
        warmup_out_dir = os.path.join(output_dir, "warmup")
        
        # Kiểm tra xem file warmup data có tồn tại không
        cord_train_path = Path(args.project_root) / cord_train
        if not cord_train_path.exists():
            print(f"Lỗi: Không tìm thấy file warmup CORD v2 tại: {cord_train_path}")
            if args.mode == "warmup":
                sys.exit(1)
            else:
                print("Bỏ qua bước Warm-up, chuyển thẳng sang Fine-tuning...")
        else:
            current_model = train_stage(
                model_name=model_name,
                train_jsonl=str(cord_train),
                val_jsonl=None, # Warm-up trên CORD v2 không cần val set phức tạp, chỉ train n epochs
                output_dir=warmup_out_dir,
                task_token=cord_task_token,
                cord_task_token=cord_task_token,
                donut_config=donut_cfg,
                project_root=args.project_root,
                is_warmup=True
            )
            
    # 2. Fine-tuning trên dữ liệu Tiếng Việt (nếu chọn finetune hoặc full)
    if args.mode in ["finetune", "full"]:
        train_jsonl = data_cfg["processed"]["train_jsonl"]
        val_jsonl = data_cfg["processed"]["val_jsonl"]
        finetune_out_dir = os.path.join(output_dir, "finetune")
        
        train_stage(
            model_name=current_model,
            train_jsonl=str(train_jsonl),
            val_jsonl=str(val_jsonl),
            output_dir=finetune_out_dir,
            task_token=task_token,
            cord_task_token=cord_task_token,
            donut_config=donut_cfg,
            project_root=args.project_root,
            is_warmup=False
        )

if __name__ == "__main__":
    main()
