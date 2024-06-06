import os
import torch
import yaml
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
# Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

# Build Model
print('==> Build the model')
model_restored = SUNet_model(opt)
model_restored.cuda()

# Training model path direction
model_dir = Train['SAVE_DIR']
os.makedirs(model_dir, exist_ok=True)

# GPU settings
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

# Log
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

# Scheduler
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs, eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# Resume (optional)
if Train['RESUME']:
    path_chk_rest = "./pretrain-model/model_bestPSNR.pth"
    checkpoint = torch.load(path_chk_rest)
    model_restored.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    for _ in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('==> Resuming Training with learning rate:', new_lr)

# Loss
L1_loss = nn.L1Loss()

# Custom Dataset
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

# DataLoaders
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CustomImageDataset(noisy_dir='./input_images', reference_dir='./reference_images', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'], shuffle=True, num_workers=0, drop_last=False)

# Training Loop
print('==> Training start: ')
best_psnr = 0
for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    model_restored.train()
    for data in tqdm(train_loader):
        input_ = data[0].cuda()
        target = data[1].cuda()
        optimizer.zero_grad()
        restored = model_restored(input_)
        loss = L1_loss(restored, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Save model periodically or based on performance
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        torch.save({'epoch': epoch, 'state_dict': model_restored.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    # Update scheduler
    scheduler.step()
    print(f"Epoch {epoch}, Loss {epoch_loss}, Learning Rate {scheduler.get_lr()[0]}")
    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

writer.close()
