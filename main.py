import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# path = './test image/frame_00738.jpg'
# src = cv.imread(path)
# 1.55 图像的非线性灰度变换：对数变换
img = cv.imread("/home/d/git_repo/Denoising_lab/test_image/frame_00738.jpg")  # flags=0 读取为灰度图像

normImg = lambda x: 255. * (x - x.min()) / (x.max() - x.min() + 1e-6)  # 归一化
fft = np.fft.fft2(img)  # 傅里叶变换
fft_shift = np.fft.fftshift(fft)  # 中心化
amp = np.abs(fft_shift)  # 傅里叶变换的频谱
amp = np.uint8(normImg(amp))  # 映射到 [0, 255]
ampLog = np.abs(np.log(1 + np.abs(fft_shift)))  # 对数变换
ampLog = np.uint8(normImg(ampLog))  # 映射到 [0, 255]

plt.figure(figsize=(9, 5))
plt.subplot(131), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.title('Original'), plt.axis('off')
plt.subplot(132), plt.imshow(amp, cmap='gray', vmin=0, vmax=255), plt.title("FFT spectrum"), plt.axis('off')
plt.subplot(133), plt.imshow(ampLog, cmap='gray', vmin=0, vmax=255), plt.title("FFT spectrum - log trans"), plt.axis(
    'off')
plt.tight_layout()
plt.show()




