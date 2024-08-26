import torch
import torch.nn as nn
import torchvision.models as models
class SELayer(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SELayer, self).__init__()
        # Adaptive average pooling to a single element
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # Fully connected layer sequence for channel-wise feature re-calibration
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),  # Reduce dimension
            nn.ReLU(inplace=True),  # Activation function
            nn.Linear(input_dim // reduction, input_dim, bias=False),  # Increase dimension
            nn.Sigmoid()  # Sigmoid activation to scale features
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # Batch size and channel count
        y = self.avg_pool(x).view(b, c)  # Global average pooling
        y = self.fc(y).view(b, c, 1, 1)  # Pass through FC layers and reshape
        return x * y.expand_as(x)  # Scale the input by the recalibrated channel-wise features

class SE_ResNet18(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.8):
        super(SE_ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Retain all layers up to the final pooling and FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze specific layers
        for param in self.features.parameters():
            param.requires_grad = False

        # SE layers for channel-wise attention
        self.se_layers = nn.ModuleList([SELayer(channel) for channel in [64, 128, 256, 512]])
        self.avgpool = resnet.avgpool
        # Replace FC layer with dropout and new FC for classification
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Dropout for regularization
            nn.Linear(resnet.fc.in_features, num_classes)  # New fully connected layer for classification
        )

    def forward(self, x):
        for i, feature in enumerate(self.features):
            x = feature(x)
            if i in [4, 5, 6, 7]:  # Add SE layer after each residual block
                x = self.se_layers[i-4](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

