import torch
import torch.nn as nn


class Multi_CNNModel(nn.Module):
    def __init__(self, dropout_rate1=0.3,dropout_rate2=0.3,dropout_rate3=0.3,dropout_rate4=0.3,dropout_rate5=0.3):
    # def __init__(self):
        super(Multi_CNNModel, self).__init__()
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.dropout_rate4 = dropout_rate4
        self.dropout_rate5 = dropout_rate5


        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate2),

            # Adding a new convolutional layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate3),
        )

        # Placeholder for the fully connected layer's input size
        # Will set its value in the forward method
        self.fc_input_size = None

        # Fully connected layers
        # Using placeholders for now, will initialize them in forward method
        self.fc = None

    def forward(self, x):
        x = self.cnn(x)

        # Dynamically compute the fc_input_size if not already done
        if not self.fc_input_size:
            self.fc_input_size = x.size(1) * x.size(2) * x.size(3)
            self.fc = nn.Sequential(
                nn.Linear(self.fc_input_size, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate4),
                nn.Linear(256, 50),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate5),
                nn.Linear(50, 3)
            ).to(x.device)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

class Bin_CNNModel(nn.Module):
    def __init__(self):
        super(Bin_CNNModel, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 100 * 100, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x






