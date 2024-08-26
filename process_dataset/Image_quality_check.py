import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 设置图像目录和图像数量
image_dir = '/Users/xiayuqing/Desktop/ALS/L40/dataset/data_augmentation/brightness_enhanced'
# num_images = 1

# 获取所有图像文件的路径
all_image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                   fname.endswith('.tif')]
# selected_image_paths = random.sample(all_image_paths, num_images)
selected_image_paths = all_image_paths
# 定义图像转换：转换为灰度图并调整大小
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
laplacian_vars = []
brightness_means = []
contrast_stds = []
snrs = []
image_var_dict = {}
image_means_dict = {}
image_stds_dict = {}
image_snrs_dict = {}
# 遍历选择的图像
for image_path in selected_image_paths:
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV加载图像为BGR格式，转换为RGB
    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil)
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # # 显示图像
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 4, 1)
    # plt.imshow(image)
    # plt.title(image_path)
    #
    # # 直方图分析
    # plt.subplot(1, 4, 2)
    # plt.hist((image_tensor.numpy() * 255).ravel(), bins=256, histtype='step', color='black')
    # plt.title('Histogram')

    # 边缘检测
    image_numpy = image_tensor.numpy().squeeze() * 255  # 转换为NumPy数组并缩放到0-255
    image_numpy = image_numpy.astype(np.uint8)  # 转换为正确的数据类型
    blurred_image = cv2.GaussianBlur(image_numpy, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 100)  # 调整阈值以适应你的图像
    # plt.subplot(1, 4, 3)
    # plt.imshow(edges, cmap='gray')
    # plt.title('Edges')

    # 频域分析
    laplacian_var = cv2.Laplacian(image_numpy, cv2.CV_64F).var()
    # plt.subplot(1, 4, 4)
    # plt.text(0.5, 0.5, f"Laplacian\nVariance:\n{laplacian_var:.2f}", ha='center', va='center', fontsize=12)
    # plt.axis('off')
    # plt.title('Laplacian Variance')
    laplacian_vars.append(laplacian_var)
    image_var_dict[image_path] = laplacian_var

    # plt.tight_layout()
    # plt.show()
    # 计算亮度平均值和对比度（标准差）
    brightness_mean = np.mean(image_gray)
    contrast_std = np.std(image_gray)
    snr = brightness_mean / contrast_std if contrast_std else 0

    brightness_means.append(brightness_mean)
    contrast_stds.append(contrast_std)
    snrs.append(snr)
    image_means_dict[image_path] = brightness_mean
    image_stds_dict[image_path] = contrast_std
    image_snrs_dict[image_path] = snr


# 根据Laplacian方差对图片路径进行排序
sorted_paths = sorted(image_var_dict.keys(), key=lambda x: image_var_dict[x])
# 获取Laplacian方差最小的两个图片路径
min_var_paths = sorted_paths[:2]
# 获取Laplacian方差最大的两个图片路径
max_var_paths = sorted_paths[-2:]
# 根据Laplacian方差对图片路径进行排序
sorted_paths_means = sorted(image_means_dict.keys(), key=lambda x: image_means_dict[x])
# 获取Laplacian方差最小的两个图片路径
min_means_paths = sorted_paths_means[:2]
# 获取Laplacian方差最大的两个图片路径
max_means_paths = sorted_paths_means[-2:]
# 根据Laplacian方差对图片路径进行排序
sorted_paths_stds = sorted(image_stds_dict.keys(), key=lambda x: image_stds_dict[x])
# 获取Laplacian方差最小的两个图片路径
min_stds_paths = sorted_paths_stds[:2]
# 获取Laplacian方差最大的两个图片路径
max_stds_paths = sorted_paths_stds[-2:]
# 根据Laplacian方差对图片路径进行排序
sorted_paths_snrs = sorted(image_snrs_dict.keys(), key=lambda x: image_snrs_dict[x])
# 获取Laplacian方差最小的两个图片路径
min_snrs_paths = sorted_paths_snrs[:2]
# 获取Laplacian方差最大的两个图片路径
max_snrs_paths = sorted_paths_snrs[-2:]


# 输出结果
# print("Laplacian方差最小的两张图片及其方差值:")
# for path in min_var_paths:
#     print(f"{path}: {image_var_dict[path]}")
#
# print("Laplacian方差最大的两张图片及其方差值:")
# for path in max_var_paths:
#     print(f"{path}: {image_var_dict[path]}")
# print("亮度平均值最小的两张图片及其方差值:")
# for path in min_means_paths:
#     print(f"{path}: {image_means_dict[path]}")
#
# print("亮度平均值最大的两张图片及其方差值:")
# for path in max_means_paths:
#     print(f"{path}: {image_means_dict[path]}")
#
# print("对比度（标准差）最小的两张图片及其方差值:")
# for path in min_stds_paths:
#     print(f"{path}: {image_stds_dict[path]}")
#
# print("对比度（标准差）最大的两张图片及其方差值:")
# for path in max_stds_paths:
#     print(f"{path}: {image_stds_dict[path]}")
#
# print("SNR最小的两张图片及其方差值:")
# for path in min_snrs_paths:
#     print(f"{path}: {image_snrs_dict[path]}")
#
# print("SNR最大的两张图片及其方差值:")
# for path in max_snrs_paths:
#     print(f"{path}: {image_snrs_dict[path]}")

# sns.violinplot(data=laplacian_vars)
# plt.title('Laplacian Variance Distribution')
# plt.xlabel('Images')
# plt.ylabel('Laplacian Variance')
#
# plt.show()
#
# sns.violinplot(data=brightness_means)
# plt.title('Brightness Mean Distribution')
# plt.xlabel('Images')
# plt.ylabel('Brightness Mean')
# plt.show()
#
# # 绘制对比度（标准差）的分布图
# sns.violinplot(data=contrast_stds)
# plt.title('Contrast (Standard Deviation) Distribution')
# plt.xlabel('Images')
# plt.ylabel('Contrast STD')
# plt.show()
#
# # 绘制SNR的分布图
# sns.violinplot(data=snrs)
# plt.title('SNR Distribution')
# plt.xlabel('Images')
# plt.ylabel('SNR')
# plt.show()

print(laplacian_vars)
print(brightness_means)
print(contrast_stds)
print(snrs)