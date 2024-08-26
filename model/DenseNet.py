import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class FineTunedResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.8, unfreeze_layers=0):
        super(FineTunedResNet, self).__init__()
        # Load a pretrained ResNet model
        self.resnet = models.resnet18(pretrained=True)

        # Freeze the earlier layers of the model
        for name, child in list(self.resnet.named_children())[:-unfreeze_layers]:  # Freeze all layers except the last 'unfreeze_layers'
            for param in child.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer, include Dropout
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # Apply dropout with the specified rate
            nn.Linear(num_features, num_classes)  # New fully connected layer with 'num_classes' output classes
        )

    def forward(self, x):
        # Forward pass through the modified ResNet model
        return self.resnet(x)



class SELayer(nn.Module) :
    def __init__(self, input_dim, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
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


class SE_DenseNet(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(SE_DenseNet, self).__init__()
        # Load a pre-trained DenseNet121 model
        densenet121 = models.densenet121(pretrained=True)

        # Reuse the feature extraction parts of DenseNet, planning to insert SE layers
        self.features = densenet121.features

        # Assuming the channel counts after each Dense Block are known
        # Modify these values based on the actual channel sizes of your DenseNet-121 model
        block_channels = [256, 512, 1024]  # Output channels of Dense Block 1, 2, 3

        # Insert SE layers only after the first three Dense Blocks
        self.se_layers = nn.ModuleList([SELayer(channels) for channels in block_channels])

        # Replace DenseNet's classifier with a custom one
        num_ftrs = densenet121.classifier.in_features
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Apply initial conv, norm, relu, and pool layers of DenseNet features
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        # Apply Dense Block 1 and corresponding SE layer
        x = self.features.denseblock1(x)
        x = self.se_layers[0](x)
        x = self.features.transition1(x)

        # Apply Dense Block 2 and corresponding SE layer
        x = self.features.denseblock2(x)
        x = self.se_layers[1](x)
        x = self.features.transition2(x)

        # Apply Dense Block 3 and corresponding SE layer
        x = self.features.denseblock3(x)
        x = self.se_layers[2](x)
        x = self.features.transition3(x)

        # Dense Block 4 does not have a corresponding SE layer
        x = self.features.denseblock4(x)

        # Classification step
        x = self.features.norm5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

