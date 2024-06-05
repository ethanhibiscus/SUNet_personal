import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from grayscale_dataset import GrayscaleImageDataset  # Import the dataset class
from model.SUNet import SUNet_model  # Replace this with the actual module and model import

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

# Model, Loss, Optimizer
model = YourModel()  # Replace with your actual model class
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
