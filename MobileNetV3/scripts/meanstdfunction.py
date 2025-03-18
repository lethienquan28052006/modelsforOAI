import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Thư mục chứa ảnh train
DATA_DIR = "D:/Models for OAI/MobileNetV3/data"

# Chỉ resize và chuyển về tensor (chưa normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def compute_mean_std(data_dir):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)  # Đặt num_workers=0 trên Windows

    mean = torch.zeros(3)  # RGB có 3 kênh
    std = torch.zeros(3)
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # Batch size
        images = images.view(batch_samples, 3, -1)  # Đưa về dạng (batch, channel, pixels)
        mean += images.mean(dim=[0, 2]) * batch_samples
        std += images.std(dim=[0, 2]) * batch_samples
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")

if __name__ == "__main__":
    compute_mean_std(DATA_DIR)
