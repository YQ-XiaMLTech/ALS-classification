import cv2
import os
import numpy as np
import shutil

def augment_images(image_paths, labels, output_folder):
    # 创建用于存储所有增强图像路径和更新标签的列表
    augmented_image_paths = []
    augmented_labels = []

    # 对给定的图像路径列表应用各种增强技术
    flipped_images = flip_images(image_paths, output_folder)
    rotated_images = rotate_images(image_paths, output_folder)
    brightness_images = enhance_brightness(image_paths, output_folder)
    gray_imagesdir = gray_images(image_paths, output_folder)
    gray_denoising_imagesdir = gray_denoising_images(image_paths, output_folder)
    color_denoising_imagesdir = color_denoising_images(image_paths, output_folder)
    enhanced_denoising_imagesdir = enhanced_denoising_images(image_paths, output_folder)
    scaled_images = scale_images(image_paths, output_folder, scale_factor=0.9)
    cropped_images = crop_images(image_paths, output_folder, crop_size=(10, 10))
    noisy_images = add_noise_images(image_paths, output_folder, noise_level=10)
    perspective_images = perspective_transform_images(image_paths, output_folder)

    # 合并所有增强图像路径到总列表，并复制相应的标签
    augmented_image_paths.extend(image_paths)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(flipped_images)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(rotated_images)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(brightness_images)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(gray_imagesdir)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(gray_denoising_imagesdir)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(color_denoising_imagesdir)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(enhanced_denoising_imagesdir)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(scaled_images)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(cropped_images)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(noisy_images)
    augmented_labels.extend(labels)
    augmented_image_paths.extend(perspective_images)
    augmented_labels.extend(labels)

    return augmented_image_paths, augmented_labels

def original_images(train_imagedir, output_folder):
    original_folder = os.path.join(output_folder, "original")
    if not os.path.exists(original_folder):
        os.makedirs(original_folder)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        image = cv2.imread(img_path)

        output_path = os.path.join(original_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, image)

        processed_train_imagedir.append(output_path)

    return processed_train_imagedir
def flip_images(train_imagedir, output_folder, flip_code=1):
    flip_folder = os.path.join(output_folder, "flipped")
    if not os.path.exists(flip_folder):
        os.makedirs(flip_folder)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        image = cv2.imread(img_path)
        flipped_img = cv2.flip(image, flip_code)  # 1: 水平反转, 0: 垂直反转, -1: 水平和垂直反转

        output_path = os.path.join(flip_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, flipped_img)

        processed_train_imagedir.append(output_path)

    return processed_train_imagedir


def rotate_images(train_imagedir, output_folder, angle=90):
    rotate_folder = os.path.join(output_folder, "rotated")
    if not os.path.exists(rotate_folder):
        os.makedirs(rotate_folder)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, M, (w, h))

        output_path = os.path.join(rotate_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, rotated_img)

        processed_train_imagedir.append(output_path)

    return processed_train_imagedir


def enhance_brightness(train_imagedir, output_folder, factor=1.5):
    brightness_folder = os.path.join(output_folder, "brightness_enhanced")
    if not os.path.exists(brightness_folder):
        os.makedirs(brightness_folder)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        image = cv2.imread(img_path)
        enhanced_img = cv2.convertScaleAbs(image, alpha=factor, beta=0)

        output_path = os.path.join(brightness_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, enhanced_img)

        processed_train_imagedir.append(output_path)

    return processed_train_imagedir


def gray_images(train_imagedir,output_folder):
    # 创建一个新的列表来保存处理后的图像路径
    gray_folder = os.path.join(output_folder, "gray")
    if not os.path.exists(gray_folder):
        os.makedirs(gray_folder)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        # 读取图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图

        # 将灰度图像转换为三个通道的灰度图像
        image_3channel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 保存图像到指定的文件夹
        output_path = os.path.join(gray_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, image_3channel)

        # 将处理后的图像路径添加到新的列表中
        processed_train_imagedir.append(output_path)

    return processed_train_imagedir


def gray_denoising_images(train_imagedir,output_folder):
    # 创建一个新的列表来保存处理后的图像路径
    gray_denoising = os.path.join(output_folder, "gray_denoising")
    if not os.path.exists(gray_denoising):
        os.makedirs(gray_denoising)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        # 读取图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
        edges = cv2.Canny(image, 50, 150)
        # 对边缘进行膨胀，确保边缘及其内部区域不被处理
        dilated_edges = cv2.dilate(edges, None, iterations=2)
        # 创建二值掩膜，边缘及以内的区域的值为0，边缘以外的区域的值为255
        mask = cv2.threshold(dilated_edges, 127, 255, cv2.THRESH_BINARY_INV)[1]
        # 对图像进行去噪
        denoised_image = cv2.bilateralFilter(image, d=15, sigmaColor=30, sigmaSpace=75)
        # 使用掩膜结合原图像的边缘及以内的区域和去噪后的图像的边缘以外的部分
        result = cv2.bitwise_and(denoised_image, mask) + cv2.bitwise_and(image, cv2.bitwise_not(mask))
        # 将灰度图像转换为三个通道的灰度图像
        image_3channel = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # 保存图像到指定的文件夹
        output_path = os.path.join(gray_denoising, os.path.basename(img_path))
        cv2.imwrite(output_path, image_3channel)

        # 将处理后的图像路径添加到新的列表中
        processed_train_imagedir.append(output_path)

    return processed_train_imagedir

def color_denoising_images(train_imagedir,output_folder):
    # 创建一个新的列表来保存处理后的图像路径
    color_denoising = os.path.join(output_folder, "color_denoising")
    if not os.path.exists(color_denoising):
        os.makedirs(color_denoising)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        # 读取图像
        image_color = cv2.imread(img_path)
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        # 使用Canny边缘检测
        edges = cv2.Canny(image_gray, 50, 150)

        # 对边缘进行膨胀，确保边缘及其内部区域不被处理
        dilated_edges = cv2.dilate(edges, None, iterations=2)

        # 创建二值掩膜，边缘及以内的区域的值为0，边缘以外的区域的值为255
        mask = cv2.threshold(dilated_edges, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # 对彩色图像进行去噪
        denoised_image = cv2.bilateralFilter(image_color, d=15, sigmaColor=30, sigmaSpace=75)

        # 使用掩膜结合原彩色图像的边缘及以内的区域和去噪后的彩色图像的边缘以外的部分
        result = cv2.bitwise_and(denoised_image, denoised_image, mask=mask) + cv2.bitwise_and(image_color, image_color,
                                                                                              mask=cv2.bitwise_not(
                                                                                                  mask))

        # 保存图像到指定的文件夹
        output_path = os.path.join(color_denoising, os.path.basename(img_path))
        cv2.imwrite(output_path, result)

        # 将处理后的图像路径添加到新的列表中
        processed_train_imagedir.append(output_path)

    return processed_train_imagedir

def enhanced_denoising_images(train_imagedir,output_folder):
    # 创建一个新的列表来保存处理后的图像路径
    enhanced_denoising = os.path.join(output_folder, "enhanced_denoising")
    if not os.path.exists(enhanced_denoising):
        os.makedirs(enhanced_denoising)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        # 读取图像
        image_color = cv2.imread(img_path)
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        # 使用Canny边缘检测
        edges = cv2.Canny(image_gray, 50, 150)

        # 对边缘进行膨胀，确保边缘及其内部区域不被处理
        dilated_edges = cv2.dilate(edges, None, iterations=2)

        # 创建二值掩膜，边缘及以内的区域的值为0，边缘以外的区域的值为255
        mask = cv2.threshold(dilated_edges, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # 对彩色图像进行去噪
        denoised_image = cv2.bilateralFilter(image_color, d=15, sigmaColor=30, sigmaSpace=75)

        # 使用掩膜结合原彩色图像的边缘及以内的区域和去噪后的彩色图像的边缘以外的部分
        result = cv2.bitwise_and(denoised_image, denoised_image, mask=mask) + cv2.bitwise_and(image_color, image_color,
                                                                                              mask=cv2.bitwise_not(
                                                                                                  mask))
        # 转换到YUV颜色空间
        yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)

        # 对Y通道进行直方图均衡化
        # yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        # 使用CLAHE进行自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])

        # 转换回BGR颜色空间
        enhanced_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        # 保存图像到指定的文件夹
        output_path = os.path.join(enhanced_denoising, os.path.basename(img_path))
        cv2.imwrite(output_path, enhanced_image)

        # 将处理后的图像路径添加到新的列表中
        processed_train_imagedir.append(output_path)

    return processed_train_imagedir



def scale_images(train_imagedir, output_folder, scale_factor=0.9):
    scale_folder = os.path.join(output_folder, "scaled")
    if not os.path.exists(scale_folder):
        os.makedirs(scale_folder)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        # Scaling the image
        new_h, new_w = int(scale_factor * h), int(scale_factor * w)
        scaled_img = cv2.resize(image, (new_w, new_h))

        output_path = os.path.join(scale_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, scaled_img)

        processed_train_imagedir.append(output_path)

    return processed_train_imagedir

def crop_images(train_imagedir, output_folder, crop_size=(10, 10)):
    crop_folder = os.path.join(output_folder, "cropped")
    if not os.path.exists(crop_folder):
        os.makedirs(crop_folder)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        # Cropping the image
        start_row, start_col = crop_size
        cropped_img = image[start_row:h - start_row, start_col:w - start_col]

        output_path = os.path.join(crop_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, cropped_img)

        processed_train_imagedir.append(output_path)

    return processed_train_imagedir

def add_noise_images(train_imagedir, output_folder, noise_level=10):
    noise_folder = os.path.join(output_folder, "noisy")
    if not os.path.exists(noise_folder):
        os.makedirs(noise_folder)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        image = cv2.imread(img_path)

        # Adding noise to the image
        noise = np.random.randint(0, noise_level, image.shape, dtype='uint8')
        noisy_img = cv2.add(image, noise)

        output_path = os.path.join(noise_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, noisy_img)

        processed_train_imagedir.append(output_path)

    return processed_train_imagedir

def perspective_transform_images(train_imagedir, output_folder):
    perspective_folder = os.path.join(output_folder, "perspective_transformed")
    if not os.path.exists(perspective_folder):
        os.makedirs(perspective_folder)

    processed_train_imagedir = []

    for img_path in train_imagedir:
        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        # Perspective transformation
        src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        dst_points = src_points + np.random.normal(0, 5, src_points.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_img = cv2.warpPerspective(image, M, (w, h))

        output_path = os.path.join(perspective_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, warped_img)

        processed_train_imagedir.append(output_path)

    return processed_train_imagedir






