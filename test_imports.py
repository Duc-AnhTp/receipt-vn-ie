print("1. Importing torch...")
import torch
print("Torch imported successfully. Version:", torch.__version__, "CUDA:", torch.cuda.is_available())

print("2. Importing standard modules...")
import os
import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
print("Standard modules imported.")

print("3. Importing transformers...")
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
print("Transformers imported.")

print("4. Importing evaluate...")
import evaluate
print("Evaluate imported.")

print("5. Importing local modules...")
try:
    from receipt_ie.models.layoutxlm_model import setup_layoutxlm_model_and_tokenizer
    print("layoutxlm_model imported.")
except Exception as e:
    print("Failed to import layoutxlm_model:", e)

try:
    from receipt_ie.models.layoutxlm_dataset import LayoutXLMDataset, layoutxlm_collate_fn
    print("layoutxlm_dataset imported.")
except Exception as e:
    print("Failed to import layoutxlm_dataset:", e)

try:
    from receipt_ie.data.schemas import ID2LABEL
    print("schemas imported.")
except Exception as e:
    print("Failed to import schemas:", e)

print("Done! Everything imported successfully.")
