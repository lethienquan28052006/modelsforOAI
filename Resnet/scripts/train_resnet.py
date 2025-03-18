import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import os

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Lật ngang ảnh
    transforms.RandomRotation(15),  # Xoay ảnh trong khoảng [-15, 15] độ
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Thay đổi độ sáng, tương phản
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
train_data = datasets.ImageFolder(root="D:/Olympic AI 2 - resnet/data", transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Load ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 lớp tương ứng với 4 loại nấm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Early Stopping
best_acc = 0.0
patience = 100  # Số epoch tối đa không cải thiện trước khi dừng
counter = 0

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    lr_scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")
    
    # Kiểm tra Early Stopping
    if acc > best_acc:
        best_acc = acc
        counter = 0
        torch.save(model.state_dict(), "D:/Olympic AI 2 - resnet/models/resnet_mushroom.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered! Training stopped.")
            break

print("✅ Mô hình đã được lưu với độ chính xác cao nhất!")