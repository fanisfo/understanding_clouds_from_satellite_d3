import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset_loading import LoadData
from unetmcl import UNetMultiClass

label_dict = {
    "Fish": 1,
    "Flower": 2,
    "Gravel": 3,
    "Sugar":4
}


train_transform = transforms.Compose([
    transforms.ToTensor()
])

# Initialize dataset
train_dataset = LoadData(label_dict = label_dict, transformer = train_transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetMultiClass(in_channels=3, out_channels=1+len(label_dict)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Number of epochs
num_epochs = 10  # Adjust based on your needs

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for images, masks in train_loader:
        # Move data to device
        images = images.to(device)
        masks = masks.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print loss for the epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
