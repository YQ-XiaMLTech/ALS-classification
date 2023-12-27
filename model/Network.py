
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FineTunedResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.147):
        super(FineTunedResNet, self).__init__()
        # 加载预训练的ResNet模型
        self.resnet = models.resnet18(pretrained=True)

        # 冻结模型的前面层
        for name, child in list(self.resnet.named_children())[:-4]:  # 冻结除最后4个模块之外的所有层
            for param in child.parameters():
                param.requires_grad = False

        # 替换最后的全连接层，加入Dropout
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)



class FineTunedEfficientNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.147):
        super(FineTunedEfficientNet, self).__init__()
        # 加载预训练的EfficientNet模型
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

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
        return self.efficientnet(x)


