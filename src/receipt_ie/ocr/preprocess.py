import cv2
import numpy as np
from PIL import Image

def rectify_document(image: Image.Image) -> Image.Image:
    """
    Tìm viền hóa đơn và tự động căn thẳng bằng phép biến đổi phối cảnh (Perspective Transform).
    Nếu không phát hiện được viền hợp lệ có 4 góc, trả về ảnh gốc.
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Lọc nhiễu bằng Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Phát hiện cạnh bằng Canny
    edged = cv2.Canny(blurred, 50, 150)
    
    # Tìm các đường viền
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
        
    # Lấy đường viền lớn nhất
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        # Xấp xỉ đa giác để tìm 4 góc
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            # Sắp xếp các điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            
            # Tính toán kích thước ảnh đầu ra
            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))
            
            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))
            
            # Kiểm tra kích thước tối thiểu để tránh warp ra ảnh siêu nhỏ
            if max_width < 100 or max_height < 100:
                continue
                
            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype="float32")
            
            # Biến đổi phối cảnh (Warp Perspective)
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_cv, M, (max_width, max_height))
            
            return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            
    return image

def binarize_image(image: Image.Image) -> Image.Image:
    """
    Áp dụng Adaptive Thresholding để nhị phân hóa làm nét chữ hóa đơn.
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Nhị phân hóa thích ứng
    thresh = cv2.adaptiveThreshold(
        img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh).convert("RGB")
