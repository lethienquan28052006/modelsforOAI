import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import pandas as pd

# Định nghĩa transform (phải giống với khi train)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 lớp nấm
model.load_state_dict(torch.load("D:/Olympic AI 2 - resnet/models/resnet_mushroom.pth", map_location=device))
model.to(device)
model.eval()

# Đáp án đúng theo thứ tự đã cho
ground_truth = [1] * 50 + [2] * 50 + [3] * 50 + [0] * 50

# Đọc ảnh từ thư mục test
test_dir = "./test_data/"
image_files = sorted(os.listdir(test_dir))  # Đảm bảo đúng thứ tự ảnh
predictions = []

correct = 0
total = len(image_files)

with torch.no_grad():
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(test_dir, image_file)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
        
        predictions.append((image_file, predicted_class))
        
        # So sánh với ground_truth
        if predicted_class == ground_truth[i]:
            correct += 1

# Tính accuracy
accuracy = correct / total * 100
print(f"🔥 Accuracy: {accuracy:.2f}% ({correct}/{total})")

# Xuất file CSV
df = pd.DataFrame(predictions, columns=["id", "type"])
df.to_csv("test_predictions.csv", index=False)
print("✅ File test_predictions.csv đã được tạo!")
