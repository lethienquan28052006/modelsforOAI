import torch
import torch.nn as nn
from torchvision.models import resnet18, efficientnet_b0, mobilenet_v3_large

class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNN, self).__init__()
        
        # 📌 ResNet18 (512 features)
        self.resnet = resnet18(weights="IMAGENET1K_V1")
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Bỏ fully connected layer cuối
        self.resnet_pool = nn.AdaptiveAvgPool2d(1)  # Áp dụng pooling để giảm kích thước

        # 📌 EfficientNet-B0 (1280 features)
        self.efficientnet = efficientnet_b0(weights="IMAGENET1K_V1")
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-2])
        self.efficientnet_pool = nn.AdaptiveAvgPool2d(1)

        # 📌 MobileNetV3-Large (960 features)
        self.mobilenet = mobilenet_v3_large(weights="IMAGENET1K_V1")
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-2])
        self.mobilenet_pool = nn.AdaptiveAvgPool2d(1)

        # 📌 Fully connected layer cuối cùng
        self.fc = nn.Sequential(
            nn.Linear(512 + 1280 + 960, 1024),  # Tổng số features từ 3 backbone
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # 📌 ResNet18 features
        resnet_features = self.resnet(x)
        resnet_features = self.resnet_pool(resnet_features).view(resnet_features.size(0), -1)

        # 📌 EfficientNet-B0 features
        efficientnet_features = self.efficientnet(x)
        efficientnet_features = self.efficientnet_pool(efficientnet_features).view(efficientnet_features.size(0), -1)

        # 📌 MobileNetV3-Large features
        mobilenet_features = self.mobilenet(x)
        mobilenet_features = self.mobilenet_pool(mobilenet_features).view(mobilenet_features.size(0), -1)

        # 📌 Ghép tất cả features lại
        combined_features = torch.cat((resnet_features, efficientnet_features, mobilenet_features), dim=1)

        # 📌 Dự đoán đầu ra
        return self.fc(combined_features)