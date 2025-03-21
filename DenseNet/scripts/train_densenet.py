import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import densenet121
from torch.utils.data import DataLoader, random_split
import os

# ======= Tham số =======
DATA_DIR = "D:/Models for OAI/DenseNet/data"
BATCH_SIZE = 64
NUM_EPOCHS = 100
VALID_RATIO = 0.15  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Transform (Không có Data Augmentation) =======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4397, 0.3948, 0.3603], std=[0.2106, 0.1945, 0.1902])
])

# ======= Load Dataset =======
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_train = int((1 - VALID_RATIO) * len(dataset))
num_val = len(dataset) - num_train
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

# ======= Load DenseNet121 =======
model = densenet121(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False  # Đóng băng tất cả tham số

# Chỉ mở khóa fine-tune từ layer cuối cùng (classifier và layer gần cuối)
for param in model.features[-4:].parameters():
    param.requires_grad = True  # Chỉ fine-tune 4 block cuối cùng

# Thay thế classifier
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)

model = model.to(DEVICE)

# ======= Loss & Optimizer =======
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ======= Training Loop =======
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    scheduler.step()

    # ======= Validation =======
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    # Lưu model tốt nhất
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "D:/Models for OAI/DenseNet/models/densenet121_best.pth")
        print("✅ Model updated successfully!")

print("✅ Training complete!")
