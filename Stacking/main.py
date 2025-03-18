import torch
import torch.optim as optim
import torch.nn as nn
from scripts.dataset import get_data_loaders
from scripts.model import MushroomNet

# Khởi tạo mô hình
train_loader, num_classes = get_data_loaders("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MushroomNet(num_classes).to(device)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Lưu model
torch.save(model.state_dict(), "models/mushroom_model.pth")
print("Model đã được lưu!")