import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 设置图像目录和图像数量
image_dir = 'autodl-tmp/ALS_classification/dataset/AptamerROIs020623'
num_images = 20

# 获取所有图像文件的路径
all_image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                   fname.endswith('.tif')]
selected_image_paths = random.sample(all_image_paths, num_images)

# 定义图像转换：转换为灰度图并调整大小
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 遍历选择的图像
for image_path in selected_image_paths:
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV加载图像为BGR格式，转换为RGB
    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil)

    # 显示图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')

    # 直方图分析
    plt.subplot(1, 4, 2)
    plt.hist((image_tensor.numpy() * 255).ravel(), bins=256, histtype='step', color='black')
    plt.title('Histogram')

    # 边缘检测
    image_numpy = image_tensor.numpy().squeeze() * 255  # 转换为NumPy数组并缩放到0-255
    image_numpy = image_numpy.astype(np.uint8)  # 转换为正确的数据类型
    blurred_image = cv2.GaussianBlur(image_numpy, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 100)  # 调整阈值以适应你的图像
    plt.subplot(1, 4, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')

    # 频域分析
    laplacian_var = cv2.Laplacian(image_numpy, cv2.CV_64F).var()
    plt.subplot(1, 4, 4)
    plt.text(0.5, 0.5, f"Laplacian\nVariance:\n{laplacian_var:.2f}", ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.title('Laplacian Variance')

    plt.tight_layout()
    plt.show()
