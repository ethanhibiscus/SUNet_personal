import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import os
import yaml
from tqdm import tqdm
from collections import OrderedDict
from model.SUNet_detail import SUNet


class SUNet_model(nn.Module):
    def __init__(self, config):
        super(SUNet_model, self).__init__()
        self.config = config
        self.swin_unet = SUNet(img_size=config['SWINUNET']['IMG_SIZE'],
                               patch_size=config['SWINUNET']['PATCH_SIZE'],
                               in_chans=1,  # Changed from 3 to 1 for grayscale images
                               out_chans=1,  # Changed from 3 to 1 for grayscale images
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


def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model_state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)


if __name__ == '__main__':
    # Load yaml configuration file
    with open('training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    # Paths
    image_dir = './input_images/'
    pretrained_weights_path = './pretrained_weights.pth'

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(image_dir=image_dir, transform=transform)
    val_size = 100
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=Train['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Train['BATCH_SIZE'], shuffle=False)

    # Model
    model = SUNet_model(opt)

    # Load pretrained weights
    load_checkpoint(model, pretrained_weights_path)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=OPT['LR'])

    # Training loop
    for epoch in range(Train['EPOCHS']):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(val_loader)}")
