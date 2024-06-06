import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from model.SUNet import SUNet_model
import math
from tqdm import tqdm
import yaml
from skimage import img_as_ubyte

# Load configuration
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)

# Argument parsing
parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default='./input_images/', type=str, help='Input images')
parser.add_argument('--window_size', default=8, type=int, help='window size')
parser.add_argument('--result_dir', default='./finetuned_output_results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrain-model/model_bestPSNR.pth', type=str, help='Path to weights')
args = parser.parse_args()

# Function definitions
def overlapped_square(timg, kernel=256, stride=128):
    patch_images = []
    b, c, h, w = timg.size()
    X = int(math.ceil(max(h, w) / float(kernel)) * kernel)
    img = torch.zeros(1, 3, X, X).type_as(timg)
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    patch = img.unfold(3, kernel, stride).unfold(2, kernel, stride)
    patch = patch.contiguous().view(b, c, -1, kernel, kernel)
    patch = patch.permute(2, 0, 1, 4, 3)

    for each in range(len(patch)):
        patch_images.append(patch[each])

    return patch_images, mask, X

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                  + glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

model = SUNet_model(opt)
model.cuda()
load_checkpoint(model, args.weights)
model.eval()

print('restoring images...')

stride = args.stride
window_size = args.window_size

for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()
    
    # Get original dimensions
    original_height, original_width = img.size[1], img.size[0]

    # Resize image while maintaining aspect ratio
    max_dim = max(original_height, original_width)
    scale_factor = 256 / max_dim
    new_height = int(scale_factor * original_height)
    new_width = int(scale_factor * original_width)
    resized_img = TF.resize(img, (new_height, new_width))
    input_ = TF.to_tensor(resized_img).unsqueeze(0).cuda()
    
    # Pad the image to be divisible by the window size
    pad_h = (window_size - new_height % window_size) % window_size
    pad_w = (window_size - new_width % window_size) % window_size
    input_ = F.pad(input_, (0, pad_w, 0, pad_h), 'reflect')

    with torch.no_grad():
        square_input_, mask, max_wh = overlapped_square(input_.cuda(), kernel=window_size, stride=stride)
        output_patch = torch.zeros(square_input_[0].shape).type_as(square_input_[0])
        for i, data in enumerate(square_input_):
            restored = model(square_input_[i])
            if i == 0:
                output_patch += restored
            else:
                output_patch = torch.cat([output_patch, restored], dim=0)

        B, C, PH, PW = output_patch.shape
        weight = torch.ones(B, C, PH, PH).type_as(output_patch)
        patch = output_patch.contiguous().view(B, C, -1, window_size * window_size)
        patch = patch.permute(2, 1, 3, 0)
        patch = patch.contiguous().view(1, C * window_size * window_size, -1)
        weight_mask = weight.contiguous().view(B, C, -1, window_size * window_size)
        weight_mask = weight_mask.permute(2, 1, 3, 0)
        weight_mask = weight_mask.contiguous().view(1, C * window_size * window_size, -1)
        restored = F.fold(patch, output_size=(max_wh, max_wh), kernel_size=window_size, stride=stride)
        we_mk = F.fold(weight_mask, output_size=(max_wh, max_wh), kernel_size=window_size, stride=stride)
        restored /= we_mk
        restored = torch.masked_select(restored, mask.bool()).reshape(input_.shape)
        restored = torch.clamp(restored, 0, 1)

    # Resize the restored image back to the original dimensions
    restored = TF.resize(restored, (original_height, original_width))

    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])
    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), restored)

print(f"Files saved at {out_dir}")
