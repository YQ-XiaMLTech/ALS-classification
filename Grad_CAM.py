import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from model.SE_ResNet18 import SE_ResNet18
from process_dataset import pre_data
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


# 模型推理和Grad-CAM相关的函数
def apply_gradcam(input_tensor, model, target_layer, target_category):
    # 存储激活和梯度
    activations = None
    gradients = None

    # 钩子函数
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    # 注册钩子
    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_full_backward_hook(backward_hook)

    # 前向传播
    output = model(input_tensor.unsqueeze(0))
    model.zero_grad()

    # 获取目标类别
    if target_category is None:
        target_category = output.argmax(dim=1)

    # 目标类别的得分
    score = output[:, target_category].squeeze()

    # 反向传播
    score.backward()

    # 梯度池化
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # 权重激活图层
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    # 清除钩子
    hook_forward.remove()
    hook_backward.remove()

    return heatmap


# 主程序
def main():
    # 模型路径
    model_path = "/Users/xiayuqing/Desktop/ALS/L40/saves/model_fullbest.pth"

    # 图像路径
    # img_path = "dataset/AptamerROIs020623/4.tif"
    # dataset_path = "dataset/AptamerROIs020623"
    # target_category = 2
    # Original ='DenseNet121+SE:Control'
    # Grad_CAM = 'DenseNet121+SE:Discordant'

    # img_path = "dataset/AptamerROIs020623/1.tif"
    # dataset_path = "dataset/AptamerROIs020623"
    # target_category = 2
    # Original = 'DenseNet121+SE:Concordant'
    # Grad_CAM = 'DenseNet121+SE:Discordant'

    img_path = "dataset/AptamerROIs020623/3.tif"
    dataset_path = "dataset/AptamerROIs020623"
    target_category = 2
    Original = 'DenseNet121:Discordant'
    Grad_CAM = 'DenseNet121+SE:Discordant'

    # 加载模型
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # model = SE_ResNet18(num_classes=3)
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 预处理
    mean, std = pre_data.compute_mean_std(dataset_path)
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 加载和变换图像
    img_original=Image.open(img_path)
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)

    # 应用Grad-CAM
    target_layer = model.features[-1]  # 假定为最后一个特征提取层
    # target_layer = model.cnn[-5]  # CNN
    # target_layer = model.resnet.layer4[-1] # ResNet
    # target_layer = model.features.norm5 #DenseNet

    heatmap = apply_gradcam(img_tensor, model, target_layer, target_category=target_category)

    img_pil = to_pil_image(img_tensor)
    heatmap_detached = heatmap.squeeze().cpu().detach()
    # 将热力图从tensor转换为PIL图像，并调整大小以匹配原图
    overlay = to_pil_image(heatmap_detached, mode='F').resize(img_pil.size, PIL.Image.BICUBIC)
    # overlay = to_pil_image(heatmap.squeeze(), mode='F').resize(img_pil.size, PIL.Image.BICUBIC)

    # 将PIL图像转换为numpy数组，以便使用matplotlib显示
    overlay_np = np.array(overlay)

    # 应用颜色映射到热力图
    overlay_colormap = cm.jet(overlay_np)  # 使用jet颜色映射

    # 将颜色映射后的热力图（RGBA）转换为RGB，丢弃A通道
    overlay_colormap_rgb = (overlay_colormap[..., :3] * 255).astype(np.uint8)

    # 创建一个新的matplotlib图像窗口
    plt.figure(figsize=(10, 5))

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(img_original)
    plt.title(Original)
    plt.axis('off')  # 移除坐标轴

    # 显示热力图
    plt.subplot(1, 2, 2)
    plt.imshow(img_original)
    plt.imshow(overlay_colormap_rgb, alpha=0.5)  # alpha参数控制热力图的透明度
    plt.title(Grad_CAM)
    plt.axis('off')  # 移除坐标轴

    plt.show()


if __name__ == '__main__':
    main()
