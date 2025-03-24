import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
from hybrid_model import HybridCNN  # Import mô hình đã train

# 📌 Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Định nghĩa transform (giống khi train)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4397, 0.3948, 0.3603], std=[0.2106, 0.1945, 0.1902])
])

# 📌 Cấu hình đường dẫn
MODEL_PATH = "D:/Models for OAI/HybridModel/models/hybrid_triple.pth"
TEST_DIR = "./test_data/"

# 📌 Load model HybridCNN
num_classes = 4  # Bạn cần chỉnh lại nếu số class khác
model = HybridCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 📌 Đáp án đúng theo thứ tự ảnh
ground_truth = [1] * 50 + [2] * 50 + [3] * 50 + [0] * 50

# 📌 Đọc danh sách ảnh trong thư mục test
image_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith((".jpg", ".png", ".jpeg"))])
predictions = []

correct = 0
total = len(image_files)

with torch.no_grad():
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(TEST_DIR, image_file)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
        
        predictions.append((image_file, predicted_class))
        
        # So sánh với ground_truth nếu có đủ dữ liệu
        if i < len(ground_truth) and predicted_class == ground_truth[i]:
            correct += 1

        # Hiển thị tiến trình
        print(f"[{i+1}/{total}] 🔍 {image_file} → Predict: {predicted_class}")

# 📌 Tính accuracy
accuracy = correct / total * 100
print(f"\n🔥 Accuracy: {accuracy:.2f}% ({correct}/{total})")

# 📌 Xuất file CSV
df = pd.DataFrame(predictions, columns=["id", "type"])
df.to_csv("test_predictions_hybrid.csv", index=False)
print("✅ File test_predictions_hybrid.csv đã được tạo!")
