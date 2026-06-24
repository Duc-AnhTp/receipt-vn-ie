# Hướng dẫn sử dụng báo cáo LaTeX

## Upload lên Overleaf

1. **Nén thư mục `report/`** thành file ZIP
2. Truy cập [Overleaf](https://www.overleaf.com) → **New Project** → **Upload Project**
3. Upload file ZIP vừa tạo

## Cấu hình Compiler

> **QUAN TRỌNG**: Phải chuyển compiler sang **XeLaTeX** để hỗ trợ tiếng Việt.

1. Trên Overleaf: **Menu** (góc trái trên) → **Settings**
2. **Compiler**: chọn **XeLaTeX**
3. **Main document**: chọn `main.tex`
4. Nhấn **Recompile**

## Cấu trúc file

```
report/
├── main.tex              ← File chính (compile file này)
├── preamble.tex           ← Packages, settings, commands
├── references.bib         ← Tài liệu tham khảo (BibTeX)
├── .latexmkrc             ← Config cho XeLaTeX
├── chapters/
│   ├── 00_abstract.tex    ← Tóm tắt
│   ├── 01_introduction.tex
│   ├── 02_theory.tex      ← Chương lớn nhất (~15 trang)
│   ├── 03_dataset.tex
│   ├── 04_preprocessing.tex
│   ├── 05_methodology.tex
│   ├── 06_experiments.tex
│   ├── 07_results.tex     ← Chương quan trọng nhất
│   ├── 08_conclusion.tex
│   └── appendix.tex
├── generated/             ← Bảng kết quả LaTeX đã chốt cho báo cáo
└── figures/
    ├── huce_logo.jpg
    ├── donut_architecture.jpg
    ├── pipeline_overview.png
    ├── em_present_only_comparison.png
    └── gradio_demo.png
```

## Trạng thái hiện tại

Toàn bộ nội dung 10 chương đã được rà soát, thống nhất lại với artifact thực nghiệm và sẵn sàng biên dịch bằng XeLaTeX.

### Nội dung đã hoàn thiện

- [x] **Trang bìa**: GVHD ThS. Nguyễn Đình Quý, 3 SV, lớp 68CS2
- [x] **Lời cảm ơn**: Đã viết trong `main.tex`
- [x] **Tóm tắt (Abstract)**: Đã viết trong `00_abstract.tex`
- [x] **Chương 1-8**: Nội dung đầy đủ
- [x] **Phụ lục**: Cấu hình YAML, ví dụ kết quả

### Hình ảnh đang dùng trong PDF

Bản PDF hiện tại chỉ nạp các hình cần thiết cho nội dung chính:

- [x] `huce_logo.jpg` — Logo trường trên trang bìa
- [x] `donut_architecture.jpg` — Minh họa kiến trúc Donut
- [x] `pipeline_overview.png` — Sơ đồ ba pipeline
- [x] `em_present_only_comparison.png` — Biểu đồ EM present-only sinh từ metrics JSON
- [x] `gradio_demo.png` — Ảnh chụp giao diện demo từ lượt chạy thực tế

## Custom Commands

Sử dụng các lệnh custom đã định nghĩa sẵn:

| Lệnh | Ví dụ | Kết quả |
|-------|-------|---------| 
| `\field{store\_name}` | Tên trường | **`store_name`** |
| `\model{Donut}` | Tên mô hình | DONUT (small caps) |
| `\metric{EM}` | Tên metric | *EM* (italic) |
| `\token{s\_receipt\_ie}` | Task token | `<s_receipt_ie>` |
| `\gls{ocr}` | Từ viết tắt | OCR (+ giải thích lần đầu) |
| `\cite{kim2022donut}` | Trích dẫn | [1] |

## Logo HUCE

Logo HUCE đã được thay bằng ảnh chính thức (`figures/huce_logo.jpg`).
