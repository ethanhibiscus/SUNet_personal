import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import os
from model.SUNet_detail import SUNet
import yaml
from tqdm import tqdm
from collections import OrderedDict

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, image

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset
dataset = CustomDataset(image_dir='./input_images/', transform=transform)
val_size = 100
train_size = len(dataset) - val_size
val_dataset, train_dataset = random_split(dataset, [val_size, train_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define model
class SUNet_model(nn.Module):
    def __init__(self, config):
        super(SUNet_model, self).__init__()
        self.swin_unet = SUNet(img_size=config['SWINUNET']['IMG_SIZE'],
                               patch_size=config['SWINUNET']['PATCH_SIZE'],
                               in_chans=1,  # Adjust for grayscale images
                               out_chans=1,  # Adjust for grayscale images
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
        logits = self.swin_unet(x)
        return logits

# Load configuration
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)

model = SUNet_model(opt).cuda()

# Load pretrained weights
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

pretrained_weights_path = './pretrain-model/model_bestPSNR.pth'
load_checkpoint(model, pretrained_weights_path)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt['OPTIM']['LR'])

# Training loop
num_epochs = opt['TRAINING']['EPOCHS']
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / train_size
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Evaluation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= val_size
    print(f'Validation Loss: {val_loss:.4f}')

print('Training complete')
