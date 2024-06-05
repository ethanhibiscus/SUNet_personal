import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
from model.SUNet_detail import SUNet

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

class SUNet_model(nn.Module):
    def __init__(self, config):
        super(SUNet_model, self).__init__()
        self.config = config
        self.swin_unet = SUNet(img_size=config['SWINUNET']['IMG_SIZE'],
                               patch_size=config['SWINUNET']['PATCH_SIZE'],
                               in_chans=3,
                               out_chans=3,
                               embed_dim=config['SWINUNET']['EMB_DIM'],
                               depths=config['SWINUNET']['DEPTH_EN'],
                               num_heads=config['SWINUNET']['HEAD_NUM'],
                               window_size=config['SWINUNET']['WIN_SIZE'],
                               mlp_ratio=config['SWINUNET']['MLP_RATIO'],
                               qkv_bias=config['SWINUNET']['QKV_BIAS'],
                               qk_scale=config['SWINUNET']['QK_SCALE'],
                               drop_rate=config['SWINUNET']['DROP_RATE'],
                               drop_path_rate=config['SWINUNET']['DROP_PATH_RATE'],
                               ape=config['SWINUNET']['APE'],
                               patch_norm=config['SWINUNET']['PATCH_NORM'],
                               use_checkpoint=config['SWINUNET']['USE_CHECKPOINTS'])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

# Paths
noisy_dir = "./input_images"
reference_dir = "./reference_images"
pretrained_model_path = "./pretrain_model/model_bestPSNR.pth"
save_model_path = "./fine_tuned_model.pth"

# Hyperparameters
batch_size = 16
learning_rate = 1e-4
num_epochs = 10

# Transforms
transform = transforms.ToTensor()

# Dataset and DataLoader
dataset = GrayscaleImageDataset(noisy_dir, reference_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load configuration from yaml
import yaml
with open('./model/training.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Model, Loss, Optimizer
model = SUNet_model(config)  # Initialize the SUNet model with configuration
model.load_state_dict(torch.load(pretrained_model_path))
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for noisy_imgs, reference_imgs in dataloader:
        noisy_imgs = noisy_imgs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        reference_imgs = reference_imgs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, reference_imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# Save the fine-tuned model
torch.save(model.state_dict(), save_model_path)
