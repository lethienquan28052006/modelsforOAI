import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121
from PIL import Image
import os
import pandas as pd

# Thông số
MODEL_PATH = "D:/Models for OAI/DenseNet/models/densenet121_best.pth"
TEST_DIR = "D:/Models for OAI/DenseNet/test_data"
OUTPUT_FILE = "predictions.csv"
CLASS_NAMES = ["0", "1", "2", "3"]  # Tên các class

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 4)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4397, 0.3948, 0.3603], std=[0.2106, 0.1945, 0.1902])
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

truth = [1] * 50 + [2] * 50 + [3] * 50 + [0] * 50

predictions = []

score = 0
loop = 0

for img_name in os.listdir(TEST_DIR):
    if img_name.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(TEST_DIR, img_name)
        predicted_class = predict_image(img_path, model)
        img_id = os.path.splitext(img_name)[0]
        predictions.append((img_id, predicted_class))
        loop = loop + 1
        if ((int(predicted_class)) == truth[loop - 1]): score = score + 1

# Xuất kết quả ra CSV
df = pd.DataFrame(predictions, columns=["id", "type"])
df.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to {OUTPUT_FILE} with score : ", score / loop)
