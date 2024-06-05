import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class GrayscaleImageDataset(Dataset):
    def __init__(self, noisy_dir, reference_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.reference_dir = reference_dir
        self.noisy_images = sorted([img for img in os.listdir(noisy_dir) if img.endswith('.png')])
        self.reference_images = sorted([img for img in os.listdir(reference_dir) if img.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_img_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        reference_img_path = os.path.join(self.reference_dir, self.reference_images[idx])
        
        noisy_img = Image.open(noisy_img_path).convert('L')
        reference_img = Image.open(reference_img_path).convert('L')
        
        if self.transform:
            noisy_img = self.transform(noisy_img)
            reference_img = self.transform(reference_img)
        
        return noisy_img, reference_img
