import sys
import torch
print("Step 1: Imports done.", flush=True)

from receipt_ie.models.layoutxlm_model import setup_layoutxlm_model_and_tokenizer
from receipt_ie.models.layoutxlm_dataset import LayoutXLMDataset, layoutxlm_collate_fn
print("Step 2: Local imports done.", flush=True)

try:
    print("Step 3: Loading model & tokenizer...", flush=True)
    model, tokenizer = setup_layoutxlm_model_and_tokenizer(model_name="microsoft/layoutxlm-base")
    print("Step 3 completed successfully. Model loaded.", flush=True)
except Exception as e:
    print("Step 3 failed with exception:", e, flush=True)
    sys.exit(1)

try:
    print("Step 4: Initializing datasets...", flush=True)
    train_dataset = LayoutXLMDataset(
        jsonl_path="data/processed/train.jsonl",
        tokenizer=tokenizer,
        mode="ocr_cache",
        max_length=512,
        project_root="."
    )
    print(f"Step 4 completed. Train dataset size: {len(train_dataset)}", flush=True)
except Exception as e:
    print("Step 4 failed with exception:", e, flush=True)
    sys.exit(1)

print("Done with pre-check!", flush=True)
