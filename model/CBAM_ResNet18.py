import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Adaptive average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Adaptive max pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Fully connected layer to learn attention weights
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        # Sigmoid function to scale features between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply both average and max pooling followed by fc to learn channel-wise attentions
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Convolutional layer to learn spatial attentions
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute the average and max intensities for the spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMModule(nn.Module):
    def __init__(self, in_channels):
        super(CBAMModule, self).__init__()
        # Channel attention component
        self.channel_attention = ChannelAttention(in_channels)
        # Spatial attention component
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Apply channel attention followed by spatial attention
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CBAM_ResNet18(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4):
        super(CBAM_ResNet18, self).__init__()
        # Load a pre-trained ResNet18 model
        resnet = models.resnet18(pretrained=True)
        # Retain all but the final global average pooling layer and fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze the parameters of the feature layers to prevent them from being updated during training
        for param in self.features.parameters():
            param.requires_grad = False

        # Initialize CBAM modules for each block, specified for different channels
        self.cbam_layers = nn.ModuleList([CBAMModule(channel) for channel in [64, 128, 256, 512]])
        self.avgpool = resnet.avgpool
        # Replace the original fully connected layer with one that includes dropout and is tailored to the number of classes
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Apply dropout to reduce overfitting
            nn.Linear(resnet.fc.in_features, num_classes)  # New fully connected layer for classification
        )

    def forward(self, x):
        # Pass input through ResNet features, applying CBAM after specific blocks
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i in [4, 5, 6, 7]:  # Insert CBAM layers after specific residual blocks
                x = self.cbam_layers[i-4](x)
        x = self.avgpool(x)  # Apply global average pooling
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layer
        x = self.fc(x)  # Pass through the fully connected layer
        return x



