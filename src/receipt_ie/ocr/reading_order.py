"""
Thuật toán sắp xếp thứ tự đọc (Reading Order) cho các bounding boxes OCR.
Gom nhóm các bounding boxes có cùng dòng ngang và sắp xếp từ trái qua phải, trên xuống dưới.
"""

def sort_reading_order(regions: list[dict], y_threshold: int = 12) -> tuple[list[dict], list[list[dict]]]:
    """
    Sắp xếp các vùng OCR theo thứ tự đọc tự nhiên (trên xuống dưới, trái qua phải).
    
    Args:
        regions (list[dict]): Danh sách các OCR region, mỗi vùng chứa "bbox": [x0, y0, x1, y1].
        y_threshold (int): Ngưỡng dịch chuyển theo chiều dọc (pixels) để coi là cùng một dòng.
        
    Returns:
        tuple[list[dict], list[list[dict]]]:
            - list[dict]: Danh sách phẳng các regions đã sắp xếp.
            - list[list[dict]]: Danh sách các dòng, mỗi dòng chứa các regions sắp xếp từ trái qua phải.
    """
    if not regions:
        return [], []

    # Sắp xếp thô theo y0 (cạnh trên) tăng dần
    sorted_by_y = sorted(regions, key=lambda r: r["bbox"][1])
    
    lines = []
    for r in sorted_by_y:
        bbox = r["bbox"]
        y_center = (bbox[1] + bbox[3]) / 2.0
        
        # Tìm dòng phù hợp
        placed = False
        for line in lines:
            # Tính y_center trung bình của dòng
            line_y_centers = [(item["bbox"][1] + item["bbox"][3]) / 2.0 for item in line]
            avg_line_y = sum(line_y_centers) / len(line_y_centers)
            
            # Nếu chênh lệch y_center nhỏ hơn ngưỡng y_threshold thì xếp vào dòng này
            if abs(y_center - avg_line_y) <= y_threshold:
                line.append(r)
                placed = True
                break
                
        if not placed:
            # Tạo dòng mới
            lines.append([r])
            
    # Sắp xếp các phần tử trong mỗi dòng từ trái qua phải (theo x0)
    for line in lines:
        line.sort(key=lambda r: r["bbox"][0])
        
    # Sắp xếp các dòng từ trên xuống dưới theo y_center trung bình của dòng
    def get_avg_y(line):
        line_y_centers = [(item["bbox"][1] + item["bbox"][3]) / 2.0 for item in line]
        return sum(line_y_centers) / len(line_y_centers)
        
    lines.sort(key=get_avg_y)
    
    # Tạo danh sách phẳng
    flat_sorted = []
    for line in lines:
        flat_sorted.extend(line)
        
    return flat_sorted, lines
