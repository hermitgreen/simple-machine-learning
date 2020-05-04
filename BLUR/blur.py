import cv2
import numpy as np


# 高斯噪声函数
def gauss_noise(image):
    noise_sigma = 32
    temp_image = np.float64(np.copy(image))
    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
    return noisy_image


# 读原图
raw = cv2.imread("raw.jpg")
# 处理
noise = gauss_noise(raw)

cv2.imwrite("noise.jpg", noise)

img = noise

# 高斯滤波
img3g = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imwrite("gauss3x3.jpg", img3g)
img5g = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite("gauss5x5.jpg", img5g)
img7g = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imwrite("gauss7x7.jpg", img7g)

# 均值滤波
img3b = cv2.blur(img, (3, 3))
cv2.imwrite("mean3x3.jpg", img3b)
img5b = cv2.blur(img, (5, 5))
cv2.imwrite("mean5x5.jpg", img5b)
img7b = cv2.blur(img, (7, 7))
cv2.imwrite("mean7x7.jpg", img7b)

# 中值滤波
# 由于3x3和5x5这两组数据无法在float64下处理，故转为float32
temp_image = np.float32(np.copy(img))
img3m = cv2.medianBlur(temp_image, 3)
cv2.imwrite("median3x3.jpg", img3m)
img5m = cv2.medianBlur(temp_image, 5)
cv2.imwrite("median5x5.jpg", img5m)

# 由于7x7这一组数据无法在float32下处理，故转为uint8
temp_image = np.uint8(np.copy(img))
img7m = cv2.medianBlur(temp_image, 7)
cv2.imwrite("median7x7.jpg", img7m)
