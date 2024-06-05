import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Assuming CustomDataset and SUNet_model are defined elsewhere

if __name__ == '__main__':
    # Load yaml configuration file
    with open('training.yaml', 'r') as config:
        opt = yaml.safe_load(config)

    # Check and set default values if keys are missing
    Train = opt.get('TRAINING', {})
    Train.setdefault('BATCH_SIZE', 8)  # Set a default batch size
    Train.setdefault('EPOCHS', 10)  # Set a default number of epochs
    OPT = opt.get('OPTIM', {})
    OPT.setdefault('LR', 0.001)  # Set a default learning rate

    # Paths
    image_dir = './input_images/'
    pretrained_weights_path = './pretrain-model/model_bestPSNR.pth'

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
            print("Before forward pass:", inputs.shape)  # Debugging statement
            outputs = model(inputs)
            print("After forward pass:", outputs.shape)  # Debugging statement
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

    # Save the model
    torch.save(model.state_dict(), './fine_tuned_model.pth')
    print("Model saved to ./fine_tuned_model.pth")
