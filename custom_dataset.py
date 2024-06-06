import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, noisy_dir, reference_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.reference_dir = reference_dir
        self.transform = transform
        self.noisy_images = sorted([img for img in os.listdir(noisy_dir) if img.endswith('.png')])
        self.reference_images = sorted([img for img in os.listdir(reference_dir) if img.endswith('.png')])

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_img_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        reference_img_path = os.path.join(self.reference_dir, self.reference_images[idx])
        
        noisy_image = Image.open(noisy_img_path).convert("L")
        reference_image = Image.open(reference_img_path).convert("L")

        if self.transform:
            noisy_image = self.transform(noisy_image)
            reference_image = self.transform(reference_image)
        
        return noisy_image, reference_image
