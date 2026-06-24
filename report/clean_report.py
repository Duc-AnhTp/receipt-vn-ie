import os
import re
from pathlib import Path

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        # Remove lines that are purely comments
        if stripped.startswith('%'):
            continue
        
        # Replace common AI phrases & standardizing terms
        # 1. OCR-based pipeline -> pipeline dựa trên OCR
        line = line.replace('OCR-based pipeline', 'pipeline dựa trên OCR')
        # 2. OCR-free pipeline -> pipeline không dùng OCR trung gian
        line = line.replace('OCR-free pipeline', 'pipeline không dùng OCR trung gian')
        # 3. rule-based baseline -> baseline dựa trên luật
        line = line.replace('rule-based baseline', 'baseline dựa trên luật')
        # 4. Threats to Validity -> Các yếu tố ảnh hưởng đến độ tin cậy của kết quả
        line = line.replace('Threats to Validity', 'Các yếu tố ảnh hưởng đến độ tin cậy của kết quả')
        
        new_lines.append(line)
        
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

report_dir = Path(r"d:\Users\Documents\HUCE\Thi_Giac_May_Tinh\DoAnTGMT\receipt-vn-ie\report")

for root, _, files in os.walk(report_dir):
    for file in files:
        if file.endswith('.tex'):
            process_file(os.path.join(root, file))

print("Done processing .tex files.")
