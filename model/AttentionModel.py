import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_FineTunedEfficientNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.147):
        super(SE_FineTunedEfficientNet, self).__init__()
        # Load pre-trained EfficientNet model
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # Add Squeeze-and-Excitation block
        self.se_block = SEBlock(input_dim=self.efficientnet._fc.in_features)

        # Freeze the earlier layers of the model
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer with Dropout and new FC layer
        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # Extract features
        x = self.efficientnet.extract_features(x)
        # Apply Squeeze-and-Excitation block
        x = self.se_block(x)
        # Pooling and final linear layer
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet._dropout(x)
        x = self.efficientnet._fc(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(num_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CBAM_FineTunedEfficientNet(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.147):
        super(CBAM_FineTunedEfficientNet, self).__init__()
        # 加载预训练的EfficientNet模型
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # 定义CBAM模块
        num_channels = self.efficientnet._conv_head.out_channels  # 这可能需要根据模型的具体实现进行调整
        self.cbam = CBAM(num_channels)

        # 冻结模型的前面层
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # 替换最后的全连接层，加入Dropout
        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # 特征提取
        x = self.efficientnet.extract_features(x)

        # 应用CBAM模块
        x = self.cbam(x)

        # 全局平均池化和分类器
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet._dropout(x)
        x = self.efficientnet._fc(x)
        return x

