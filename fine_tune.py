import os
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model
import utils

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, noisy_dir, reference_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.reference_dir = reference_dir
        self.transform = transform
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.reference_images = sorted(os.listdir(reference_dir))

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_image_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        reference_image_path = os.path.join(self.reference_dir, self.reference_images[idx])

        noisy_image = Image.open(noisy_image_path).convert("L")  # Ensure image is grayscale
        reference_image = Image.open(reference_image_path).convert("L")  # Ensure image is grayscale

        if self.transform:
            noisy_image = self.transform(noisy_image)
            reference_image = self.transform(reference_image)

        return noisy_image, reference_image

# Define transforms
resize_transform = transforms.Resize((256, 256))  # Adjust this size to match the expected input resolution
transform = transforms.Compose([
    resize_transform,
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
    transforms.ToTensor()  # Convert PIL image to tensor
])

# Initialize dataset
dataset = CustomDataset(noisy_dir="./input_images", reference_dir="./reference_images", transform=transform)

# Split dataset: 800 for training/validation, 200 for evaluation
train_size = 800
val_size = 200
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=2)  # Reduced batch size
val_loader = DataLoader(dataset=val_dataset, batch_size=6, shuffle=False, num_workers=1)  # Reduced batch size

# Set seeds
torch.backends.cudnn.benchmark = True
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)

# Load configuration
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

# Build model
print('==> Build the model')
model_restored = SUNet_model(opt)
model_restored.cuda()

# Optimizer and scheduler
initial_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - 3, eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_cosine)
scheduler.step()

# Load pretrained model
checkpoint = torch.load("./pretrain-model/model_bestPSNR.pth")
model_restored.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Loss function
criterion = nn.L1Loss()

# Training loop
print('==> Training start: ')
best_psnr = 0

for epoch in range(1, OPT['EPOCHS'] + 1):
    model_restored.train()
    epoch_loss = 0

    for data in tqdm(train_loader):
        noisy_images, reference_images = data
        #print(f"noisy_images shape: {noisy_images.shape}")
        #print(f"reference_images shape: {reference_images.shape}")
        noisy_images, reference_images = noisy_images.cuda(), reference_images.cuda()

        optimizer.zero_grad()
        restored_images = model_restored(noisy_images)
        
        # Check the size of the tensor before reshaping
        B, C, H, W = restored_images.size()
        #print(f"Restored images size: {restored_images.size()}")
        
        loss = criterion(restored_images, reference_images)
        #print(f"loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()

    print(f"Epoch [{epoch}/{OPT['EPOCHS']}], Loss: {epoch_loss/len(train_loader):.4f}")

    # Save the latest model
    torch.save({'epoch': epoch, 'state_dict': model_restored.state_dict(), 'optimizer': optimizer.state_dict()}, './pretrain-model/model_latest.pth')

    # Evaluate on validation set
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val = []
        with torch.no_grad():
            for val_data in tqdm(val_loader):
                noisy_images, reference_images = val_data
                noisy_images, reference_images = noisy_images.cuda(), reference_images.cuda()

                restored_images = model_restored(noisy_images)
                #print(f"Validation restored images size: {restored_images.size()}")
                psnr_val.append(utils.torchPSNR(restored_images, reference_images))

        avg_psnr_val = torch.stack(psnr_val).mean().item()
        print(f"Epoch [{epoch}], Validation PSNR: {avg_psnr_val:.4f}")

        if avg_psnr_val > best_psnr:
            best_psnr = avg_psnr_val
            torch.save({'epoch': epoch, 'state_dict': model_restored.state_dict(), 'optimizer': optimizer.state_dict()}, './pretrain-model/model_bestPSNR.pth')

print("Training completed.")
