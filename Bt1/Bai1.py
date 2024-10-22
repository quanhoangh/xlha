import cv2
import numpy as np

# Đọc ảnh
image = cv2.imread('test1.jpg')

# 1. Ảnh âm tính
negative_image = cv2.bitwise_not(image)

# 2. Tăng độ tương phản (sử dụng CLAHE)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_image = clahe.apply(gray_image)

# 3. Biến đổi log
log_image = np.log1p(image.astype(np.float32))  # Sử dụng log(1 + pixel)
log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 4. Cân bằng histogram
hist_eq_image = cv2.equalizeHist(gray_image)

# Lưu các bức ảnh đã tăng cường
cv2.imwrite('negative_image.jpg', negative_image)
cv2.imwrite('contrast_image.jpg', contrast_image)
cv2.imwrite('log_image.jpg', log_image)
cv2.imwrite('hist_eq_image.jpg', hist_eq_image)

# Hiển thị các bức ảnh (tuỳ chọn)
cv2.imshow('Negative Image', negative_image)
cv2.imshow('Contrast Image', contrast_image)
cv2.imshow('Log Image', log_image)
cv2.imshow('Histogram Equalized Image', hist_eq_image)

cv2.waitKey(0)
cv2.destroyAllWindows()