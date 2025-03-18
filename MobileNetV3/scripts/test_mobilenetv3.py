import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import os
import pandas as pd

# Định nghĩa transform (phải giống khi train)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43967315554618835, 0.3947981894016266, 0.36026132106781006], std=[0.21060341596603394, 0.19447742402553558, 0.19024626910686493])
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v3_large()
num_ftrs = model.classifier[0].in_features  # Lấy số features từ MobileNetV3

# Thay đổi classifier để phù hợp với 4 lớp nấm
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(1024, 4)  # 4 lớp đầu ra
)

# Load trọng số đã train
model.load_state_dict(torch.load("D:/Models for OAI/MobileNetV3/models/mobilenetv3.pth", map_location=device))
model.to(device)
model.eval()

# Nhãn thực tế theo thứ tự ảnh
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
        
        predictions.append((image_file[0 : 3], predicted_class))
        
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
