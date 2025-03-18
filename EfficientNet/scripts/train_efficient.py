import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import efficientnet_b2
from torch.utils.data import DataLoader

# Thiết bị chạy (GPU nếu có, nếu không thì chạy CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model EfficientNet-B2 (pretrained)
model = efficientnet_b2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(1408, 4)  # 4 lớp cho 4 loại nấm
model = model.to(device)

# Fine-tune một số lớp cuối của mô hình
for param in model.features[-3:].parameters():  # Fine-tune 3 lớp cuối
    param.requires_grad = True

# Transform dữ liệu (KHÔNG CÓ Data Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder("D:/Models for OAI/EfficientNet/data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Early Stopping
minimum_loss = 1000000000
patience = 75  # Số epoch tối đa không cải thiện trước khi dừng
counter = 0

# Training loop
num_epochs = 75
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()   
        total += labels.size(0)
    
    acc = 100 * correct / total
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")

    loss_function = running_loss / len(train_loader)

    # Kiểm tra Early Stopping
    if minimum_loss > loss_function:
        minimum_loss = loss_function
        counter = 0  # Reset nếu có cải thiện
        print("Model updated!")
        torch.save(model.state_dict(), "D:/Models for OAI/EfficientNet/models/best_efficientnet_b2.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

print("Training complete! Model saved as best_efficientnet_b2.pth")
