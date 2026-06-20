import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_metrics_comparison():
    print("Plotting metrics comparison...")
    
    # Read metrics
    paths = {
        "Baseline": "outputs/metrics/baseline_metrics.json",
        "LayoutXLM": "outputs/metrics/layoutxlm_metrics.json",
        "Donut": "outputs/metrics/donut_metrics.json"
    }
    
    data = {}
    for model_name, path in paths.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data[model_name] = json.load(f)
        else:
            print(f"Error: Metrics file for {model_name} not found at {path}")
            return
            
    fields = ["store_name", "date", "total", "address", "macro"]
    models = ["Baseline", "LayoutXLM", "Donut"]
    
    # 1. EM Comparison Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(fields))
    width = 0.25
    
    for i, model in enumerate(models):
        em_values = [data[model][f]["EM"] * 100 for f in fields]
        ax.bar(x + i*width - width/2, em_values, width, label=model)
        
    ax.set_ylabel('Exact Match (%)')
    ax.set_title('Exact Match (EM) Comparison between models')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([f.upper() for f in fields])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/em_comparison.png", dpi=150)
    plt.close()
    
    # 2. NES Comparison Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        nes_values = [data[model][f]["NES"] * 100 for f in fields]
        ax.bar(x + i*width - width/2, nes_values, width, label=model)
        
    ax.set_ylabel('Edit Similarity (%)')
    ax.set_title('Edit Similarity (NES) Comparison between models')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([f.upper() for f in fields])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    plt.tight_layout()
    plt.savefig("outputs/plots/nes_comparison.png", dpi=150)
    plt.close()

    # Present-only EM avoids credit from correctly returning an empty value
    # when the corresponding ground-truth field is also empty.
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        em_values = [data[model]["present_only"][f]["EM"] * 100 for f in fields]
        ax.bar(x + i*width - width/2, em_values, width, label=model)
    ax.set_ylabel("Exact Match on present fields (%)")
    ax.set_title("Present-only Exact Match (EM)")
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([f.upper() for f in fields])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("outputs/plots/em_present_only_comparison.png", dpi=150)
    plt.close()
    print("Saved metrics comparison plots to outputs/plots/")

def plot_error_analysis():
    print("Plotting error analysis...")
    
    models = ["baseline", "layoutxlm", "donut"]
    
    for model in models:
        path = f"outputs/error_analysis/{model}_error_by_field.csv"
        if not os.path.exists(path):
            print(f"Error: Error analysis file not found at {path}")
            continue
            
        df = pd.read_csv(path)
        # Group by field and error_type
        error_counts = df.groupby(["field", "error_type"]).size().unstack(fill_value=0)
        
        # Ensure all error types are present
        all_errors = ["EMPTY_PRED", "FORMAT_ERROR", "OCR_MISS", "OCR_WRONG", "POSTPROCESS_BAD", "MODEL_BAD", "LABEL_BAD"]
        for err in all_errors:
            if err not in error_counts.columns:
                error_counts[err] = 0
                
        error_counts = error_counts[all_errors]
        
        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        error_counts.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        
        ax.set_ylabel('Error Count (samples)')
        ax.set_xlabel('Field')
        ax.set_title(f'Error type distribution per field - {model.upper()}')
        ax.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"outputs/plots/{model}_error_distribution.png", dpi=150)
        plt.close()
        
    print("Saved error analysis plots to outputs/plots/")

if __name__ == "__main__":
    plot_metrics_comparison()
    plot_error_analysis()
    print("=== PLOTTING COMPLETED ===")
