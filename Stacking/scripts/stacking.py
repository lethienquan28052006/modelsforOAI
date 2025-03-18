import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, resnet18, efficientnet_b0, densenet121
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def load_model(model_class, model_path, num_classes):
    model = model_class(pretrained=False)
    model_name = type(model).__name__
    
    if model_name == "MobileNetV3":
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, num_classes)
    elif model_name == "ResNet":
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "EfficientNet":
        num_features = model.classifier[1].in_features  # EfficientNet sử dụng classifier[1]
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == "DenseNet":
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(models, weights, dataloader, class_names):
    predictions = []
    for images, image_paths in dataloader:
        images = images.to(device)  # Đưa ảnh lên GPU nếu có
        weighted_outputs = None
        for model, weight in zip(models, weights):
            model = model.to(device)
            outputs = model(images) * weight  # Nhân với trọng số
            if weighted_outputs is None:
                weighted_outputs = outputs
            else:
                weighted_outputs += outputs
        weighted_outputs /= sum(weights)  # Chia cho tổng trọng số để chuẩn hóa
        final_predictions = torch.argmax(weighted_outputs, dim=1)
        
        for img_path, pred_class in zip(image_paths, final_predictions):
            img_id = os.path.basename(img_path).split('.')[0]  # Lấy ID từ tên file
            predictions.append((img_id, class_names[pred_class.item()]))
    return predictions

# Thông số
TEST_DIR = "./test_data"
MODEL_PATHS = {
    "mobilenetv3": "D:/Olympic AI - mobilenet v3/models/mobilenetv3.pth",
    "resnet18": "D:/Olympic AI 2 - resnet/models/resnet_mushroom.pth",
    "efficientnet": "D:/Olympic AI 3 - EfficientNet-B0/models/efficientnet_b0_mushroom.pth",
    "densenet": "D:/Olympic AI - dense net/models/densenet121.pth"
}
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trọng số của từng mô hình
weights = [0.2, 0.4, 0.3, 0.1]  # MobileNetV3: 30%, ResNet18: 20%, EfficientNet: 30%, DenseNet: 20%

# Chuẩn bị dữ liệu
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load các mô hình
num_classes = 4  # Số lớp trong mô hình
models = [
    load_model(mobilenet_v3_large, MODEL_PATHS["mobilenetv3"], num_classes),
    load_model(resnet18, MODEL_PATHS["resnet18"], num_classes),
    load_model(efficientnet_b0, MODEL_PATHS["efficientnet"], num_classes),
    load_model(densenet121, MODEL_PATHS["densenet"], num_classes)
]

# Dự đoán
predictions = predict(models, weights, test_loader, [str(i) for i in range(num_classes)])

# Xuất ra file CSV
df = pd.DataFrame(predictions, columns=["id", "type"])
df.to_csv("predictions.csv", index=False)
print("Dự đoán hoàn thành! File predictions.csv đã được tạo.")