import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, efficientnet_b0, mobilenet_v3_large
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

# 📌 Kiểm tra GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Cấu hình dữ liệu
DATA_DIR = "D:/Models for OAI/HybridModel/data"
BATCH_SIZE = 32  # Giảm batch size để mô hình học kỹ hơn
EPOCHS = 100  # Tăng số epochs
LEARNING_RATE = 0.001
VALID_RATIO = 0.35  # 20% dữ liệu validation
EARLY_STOP_PATIENCE = 10  # Dừng sớm nếu val acc không tăng trong 10 epochs

# 📌 Transform dữ liệu (Không có Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4397, 0.3948, 0.3603], std=[0.2106, 0.1945, 0.1902])
])

if __name__ == '__main__':
    # 📌 Load dataset
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    num_train = int((1 - VALID_RATIO) * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 📌 Số lượng lớp đầu ra
    num_classes = len(dataset.classes)
    print("Classes:", dataset.classes)

    # 🔥 Hybrid Model: EfficientNet-B0 + ResNet18 + MobileNetV3
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

    # 📌 Khởi tạo model
    model = HybridCNN(num_classes=num_classes).to(DEVICE)

    # 📌 Loss function & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 🚀 Thêm Label Smoothing để giảm overfitting
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)  # 🚀 Giảm learning rate theo Cosine Annealing

    # 📌 Training loop
    best_loss = 1e9
    early_stop_count = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # 🔍 Đánh giá trên tập validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

        # 🚀 Early Stopping: Dừng sớm nếu không cải thiện
        if best_loss > running_loss:
            best_loss = running_loss
            early_stop_count = 0
            torch.save(model.state_dict(), "D:/Models for OAI/HybridModel/models/hybrid_triple.pth")
            print("✅ Model updated successfully!")
        else:
            early_stop_count += 1
            if early_stop_count >= 30:
                print("⏹️ Early Stopping: Model không cải thiện trong 30 epochs. Dừng training!")
                break

        scheduler.step()  # 🚀 Cập nhật learning rate

    print(f"🔥 Training hoàn tất! Best Val Accuracy: {best_loss:.2f}%")
