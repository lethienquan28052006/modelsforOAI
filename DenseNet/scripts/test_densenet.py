import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121
from PIL import Image
import os
import pandas as pd

# Thông số
MODEL_PATH = "D:/Olympic AI - dense net/models/densenet121.pth"
TEST_DIR = "D:/Olympic AI - dense net/test_data"
OUTPUT_FILE = "predictions.csv"
CLASS_NAMES = ["0", "1", "2", "3"]  # Tên các class

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dự đoán ảnh
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return CLASS_NAMES[predicted.item()]

# Dự đoán tất cả ảnh trong thư mục test
predictions = []
for img_name in os.listdir(TEST_DIR):
    if img_name.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(TEST_DIR, img_name)
        predicted_class = predict_image(img_path, model)
        img_id = os.path.splitext(img_name)[0]
        predictions.append((img_id, predicted_class))

# Xuất kết quả ra CSV
df = pd.DataFrame(predictions, columns=["id", "type"])
df.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to {OUTPUT_FILE}")
