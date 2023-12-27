import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import time
import re
from collections import Counter
from config import config



def read_files(label_path, classification, type_classification):
    df = pd.read_excel(label_path, usecols=[0, 3], skiprows=1,engine='openpyxl')
    # 初始化字典来存储图像路径和标签
    images = {}
    labels = {}
    # 遍历DataFrame的每一行
    if classification == 'multi-classification':
        for index, row in df.iterrows():
            # 如果第四列（在这里是列索引1，因为我们只读取了两列）的值为'Concordant'
            if row[1] == 'Control':
                images[row[0]] = f"{row[0]}.tif"
                labels[row[0]] = 0
            elif row[1] == 'Concordant':
                images[row[0]] = f"{row[0]}.tif"
                labels[row[0]] = 1
            elif row[1] == 'Discordant':
                images[row[0]] = f"{row[0]}.tif"
                labels[row[0]] = 2
    elif classification == 'binary-classification':
        words = re.findall('[A-Z][a-z]+', type_classification)
        if len(words) == 3:  # 如果有三个单词，我们将后两个单词视为同一类
            type1 = words[0]
            type2 = words[1] + words[2]
        elif len(words) == 2:  # 如果有两个单词，直接进行分类
            type1, type2 = words
        else:
            raise ValueError("Unexpected number of words in type_classification for binary classification.")

        for index, row in df.iterrows():
            # 如果第四列（在这里是列索引1，因为我们只读取了两列）的值为'Concordant'
            if row[1] == type1:
                images[row[0]] = f"{row[0]}.tif"
                labels[row[0]] = 0
            elif row[1] == type2 or row[1] in words[1:]:  # 这里处理当有三个单词时，后两个单词都被视为同一类的情况
                images[row[0]] = f"{row[0]}.tif"
                labels[row[0]] = 1
    label_counts = Counter(labels.values())
    print(label_counts)

    return images, labels

def split_datasets(dataset_path, images, labels, train_ratio, val_ratio, test_ratio):
    # 确保train_ratio + val_ratio + test_ratio = 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    # 将字典转换为列表，保持图像和标签之间的对应关系
    image_paths = [f"{dataset_path}/{img}" for img in images.values()]
    image_labels = [lbl for lbl in labels.values()]

    # 首先，分割训练集和临时集（将用于进一步分割为验证集和测试集）random_state=1,
    train_imagedir, temp_imagedir, train_labels, temp_labels = train_test_split(
        image_paths, image_labels, test_size=1 - train_ratio, random_state=config.seed, stratify=image_labels
    )

    # 计算验证集和测试集的相对大小
    rel_val_ratio = val_ratio / (val_ratio + test_ratio)
    rel_test_ratio = test_ratio / (val_ratio + test_ratio)

    # 分割临时集为验证集和测试集
    val_imagedir, test_imagedir, val_labels, test_labels = train_test_split(
        temp_imagedir, temp_labels, test_size=rel_test_ratio, random_state=config.seed, stratify=temp_labels
    )

    return train_imagedir, train_labels, val_imagedir, val_labels, test_imagedir, test_labels

def split_datasets_notvalortest(dataset_path, images, labels, train_ratio, test_ratio):
    # 确保train_ratio + val_ratio + test_ratio = 1
    assert train_ratio + test_ratio == 1, "Ratios must sum to 1"

    # 将字典转换为列表，保持图像和标签之间的对应关系
    image_paths = [f"{dataset_path}/{img}" for img in images.values()]
    image_labels = [lbl for lbl in labels.values()]

    # 首先，分割训练集和临时集（将用于进一步分割为验证集和测试集）
    train_imagedir, test_imagedir, train_labels, test_labels = train_test_split(
        image_paths, image_labels, test_size=1 - train_ratio, random_state=1, stratify=image_labels
    )

    return train_imagedir, train_labels, test_imagedir, test_labels


def compute_mean_std(image_dir):
    img_filenames = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    per_image_Rstd = []
    per_image_Gstd = []
    per_image_Bstd = []

    for img_filename in img_filenames:
        img = cv2.imread(os.path.join(image_dir, img_filename))
        img = img / 255.0  # 将像素值缩放到[0,1]之间
        per_image_Rmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Bmean.append(np.mean(img[:, :, 2]))
        per_image_Rstd.append(np.std(img[:, :, 0]))
        per_image_Gstd.append(np.std(img[:, :, 1]))
        per_image_Bstd.append(np.std(img[:, :, 2]))

    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)

    R_std = np.mean(per_image_Rstd)
    G_std = np.mean(per_image_Gstd)
    B_std = np.mean(per_image_Bstd)

    return [R_mean, G_mean, B_mean], [R_std, G_std, B_std]

def gray_compute_mean_std(image_dir):
    img_filenames = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    per_image_mean = []
    per_image_std = []

    for img_filename in img_filenames:
        img = cv2.imread(os.path.join(image_dir, img_filename), cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
        img = img / 255.0  # 将像素值缩放到[0,1]之间
        per_image_mean.append(np.mean(img))
        per_image_std.append(np.std(img))

    mean = np.mean(per_image_mean)
    std = np.mean(per_image_std)

    return [mean], [std]




# if __name__ == "__main__":
# #     images, labels = read_files(config.label)
# #     print(images,'\n',labels)
# #     train_imagedir, train_labels, val_imagedir, val_labels, test_imagedir, test_labels = split_datasets(
# #         config.dataset, images, labels, config.train_ratio, config.val_ratio, config.test_ratio)
# #     print(train_imagedir,'\n', train_labels, '\n',val_imagedir, '\n',val_labels, '\n',test_imagedir, '\n',test_labels)
# #     print(len(train_imagedir),len(val_imagedir),len(test_imagedir))
#     # 使用你的图像数据目录替换下面的路径
#     image_dir = config.dataset
#     mean, std = compute_mean_std(image_dir)
#
#     print(f'Mean: {mean}')
#     print(f'STD: {std}')
