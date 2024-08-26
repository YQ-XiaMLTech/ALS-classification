import torch
import torchvision.models as models
from model.SE_ResNet18 import SE_ResNet18
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os.path as osp
from config import config
from process_dataset import pre_data

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/Users/xiayuqing/Desktop/ALS/L40/saves/20240322_141129_121188-multi-classification_ControlConcordantDiscordant-batch_size32-lr5e-05/model_full.pth"

# 初始化模型
num_classes = 3  # 根据你的具体情况调整
# model = SE_ResNet18(num_classes=3,dropout_rate=0.85).to(config.device)
model = torch.load(model_path, map_location=config.device)
model.eval()

img_path = "dataset/AptamerROIs020623/1.tif"
dataset_path = "dataset/AptamerROIs020623"
input_image = Image.open(img_path)
mean, std = pre_data.compute_mean_std(dataset_path)
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)  # 创建一个batch # 添加批次维度
# 前向传播，获取输出和注意力权重
with torch.no_grad():  # 不计算梯度
    output, attention_weights_list = model(input_batch)
attention_weights = attention_weights_list[-1]
main_save_path = "/Users/xiayuqing/Desktop/ALS/L40/saves"  # 调整为实际保存目录
attention_matrix = attention_weights.cpu().detach().numpy()

plt.figure(figsize=(10, 8))
plt.imshow(attention_matrix, cmap='viridis', interpolation='none')  # 选择一个好的颜色映射
plt.colorbar()
plt.title('Attention Map')
figure_attention_map = osp.join(main_save_path, "attention_map.png")
plt.savefig(figure_attention_map)
plt.close()
