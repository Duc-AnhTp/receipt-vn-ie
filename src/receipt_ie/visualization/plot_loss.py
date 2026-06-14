import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Đảm bảo terminal Windows in được tiếng Việt UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Dự phòng cho một số môi trường python cũ không có reconfigure

def plot_loss(state_path: str, output_path: str):
    """
    Đọc file trainer_state.json từ Seq2SeqTrainer và vẽ biểu đồ hàm loss (train loss và eval loss nếu có).
    """
    print(f"Đang đọc dữ liệu huấn luyện từ: {state_path}...")
    
    if not os.path.exists(state_path):
        print(f"Lỗi: Không tìm thấy file state tại: {state_path}")
        return False
        
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state_data = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file JSON: {e}")
        return False
        
    log_history = state_data.get("log_history", [])
    if not log_history:
        print("Cảnh báo: Không tìm thấy dữ liệu 'log_history' trong file state.")
        return False
        
    train_steps = []
    train_losses = []
    train_epochs = []
    
    eval_steps = []
    eval_losses = []
    eval_epochs = []
    
    for log in log_history:
        step = log.get("step")
        epoch = log.get("epoch")
        
        # Lấy train loss
        if "loss" in log:
            train_steps.append(step)
            train_losses.append(log["loss"])
            train_epochs.append(epoch)
            
        # Lấy eval loss
        if "eval_loss" in log:
            eval_steps.append(step)
            eval_losses.append(log["eval_loss"])
            eval_epochs.append(epoch)
            
    if not train_losses and not eval_losses:
        print("Không tìm thấy thông tin loss hay eval_loss trong log_history.")
        return False
        
    # Tạo thư mục đầu ra nếu chưa có
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    # Thiết lập giao diện biểu đồ
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Thiết kế màu sắc hiện đại
    color_train = "#1f77b4" # Royal Blue
    color_eval = "#d62728"  # Crimson Red
    
    # Vẽ Train Loss
    line1 = None
    if train_losses:
        line1 = ax1.plot(
            train_steps, 
            train_losses, 
            color=color_train, 
            linestyle="-", 
            linewidth=1.8, 
            label=f"Train Loss (Min: {min(train_losses):.4f})"
        )
        ax1.set_xlabel("Steps (Huấn luyện)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
        ax1.tick_params(axis="both", labelsize=9)
        
    # Vẽ Eval Loss
    line2 = None
    if eval_losses:
        line2 = ax1.plot(
            eval_steps, 
            eval_losses, 
            color=color_eval, 
            linestyle="--", 
            marker="o", 
            markersize=5, 
            linewidth=1.8, 
            label=f"Eval Loss (Min: {min(eval_losses):.4f})"
        )
        
    # Tạo tiêu đề
    plt.title("Biểu Đồ Hàm Loss Donut Trong Quá Trình Huấn Luyện", fontsize=14, fontweight="bold", pad=15)
    
    # Gộp legend
    lines = []
    if line1:
        lines += line1
    if line2:
        lines += line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, shadow=True, fontsize=10)
    
    # Thêm trục Epoch phụ ở phía trên để dễ quan sát
    if train_epochs and len(train_epochs) == len(train_steps):
        # Trục thứ hai biểu thị Epochs
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        
        # Định nghĩa các mốc hiển thị Epoch
        num_ticks = min(8, len(train_steps))
        tick_indices = [int(i * (len(train_steps) - 1) / (num_ticks - 1)) for i in range(num_ticks)] if len(train_steps) > 1 else [0]
        
        tick_steps = [train_steps[idx] for idx in tick_indices]
        tick_labels = [f"{train_epochs[idx]:.2f}" for idx in tick_indices]
        
        ax2.set_xticks(tick_steps)
        ax2.set_xticklabels(tick_labels, fontsize=9)
        ax2.set_xlabel("Epochs tương ứng", fontsize=11, fontweight="bold", labelpad=10)
        ax2.grid(False) # Tắt grid của trục phụ để tránh rối mắt
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Vẽ biểu đồ thành công! File biểu đồ được lưu tại: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ loss từ trainer_state.json")
    parser.add_argument(
        "--state_path", 
        type=str, 
        default="checkpoints/donut/receipt_ie/finetune/checkpoint-21/trainer_state.json",
        help="Đường dẫn tới file trainer_state.json chứa log huấn luyện"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="outputs/plots/donut_loss.png",
        help="Đường dẫn lưu file ảnh biểu đồ hàm loss"
    )
    args = parser.parse_args()
    
    plot_loss(args.state_path, args.output_path)

if __name__ == "__main__":
    main()
