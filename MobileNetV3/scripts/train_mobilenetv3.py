import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v3_large
from torch.utils.data import random_split, DataLoader

# Thông số cấu hình
DATA_DIR = "D:/Models for OAI/MobileNetV3/data"  # Thư mục chứa dữ liệu
BATCH_SIZE = 64  # Tăng batch size giúp ổn định gradient
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2  # 20% dữ liệu để validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chuẩn bị dữ liệu (Không có Data Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43967315554618835, 0.3947981894016266, 0.36026132106781006], std=[0.21060341596603394, 0.19447742402553558, 0.19024626910686493])
])

# Load dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_train = int((1 - VALIDATION_SPLIT) * len(dataset))
num_val = len(dataset) - num_train
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Tạo model MobileNetV3
model = mobilenet_v3_large(pretrained=True)

# Lấy đúng số lượng đặc trưng từ MobileNetV3
num_features = model.classifier[0].in_features

# Thay thế bộ phân loại cuối cùng
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1024),  # Thêm tầng ẩn để tăng hiệu suất
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 4)  # 4 lớp đầu ra cho 4 loại nấm
)

model = model.to(DEVICE)

# Khởi tạo trọng số với Xavier Uniform
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.classifier.apply(initialize_weights)

# Loss (Label Smoothing để giảm overfitting)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer AdamW (tốt hơn Adam)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning Rate Scheduler (Cosine Annealing)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=len(train_loader), epochs=EPOCHS)

# Training loop

min_loss_function = 10000000000

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient Clipping để tránh gradient exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()

    # Đánh giá trên tập validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    lr_scheduler.step()  # Cập nhật Learning Rate
    loss_function = running_loss / len(train_loader)
    if (min_loss_function > loss_function):
        min_loss_function = loss_function
        print("Model updated successfully")
        torch.save(model.state_dict(), "D:/Models for OAI/MobileNetV3/models/mobilenetv3.pth")

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

# Lưu model

print("✅ Model đã được lưu thành công!")
