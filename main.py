import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet

from dataset_loading import LoadData
from unetmcl import UNetMultiClass
from dice_loss import DiceLoss
from utils import *

label_dict = {
    "Fish": 1,
    "Flower": 2,
    "Gravel": 3,
    "Sugar":4
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetMultiClass(out_channels=len(label_dict)).to(device)
dice_loss_fn = DiceLoss(label_dict)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5

# Training
train_transform = transforms.Compose([
    transforms.Resize((350, 525)),
    transforms.ToTensor()
])

train_dataset = LoadData(
    label_dict = label_dict, 
    transformer = train_transform, 
    images_dir = 'understanding_cloud_organization/train_images',
    mask_csv_path="understanding_cloud_organization/train.csv")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, masks, _ in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images) #torch.Size([8, 4, 350, 525])

        loss = dice_loss_fn(outputs, masks)
        print(loss)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Testing
test_transform = transforms.Compose([
    transforms.Resize((350, 525)),
    transforms.ToTensor()
])

test_dataset = LoadData(
    label_dict = label_dict,
    transformer = None,
    images_dir = 'understanding_cloud_organization/test_images'
)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

model.eval()

df_result = pd.DataFrame(columns=["Image_Label","EncodedPixels"])

for images, _, img_names in test_loader:
    images = images.to(device)
    masks = masks.to(device)

    outputs = model(images)  
    
    for i in outputs.shape[0]:
        for j in outputs.shape[1]:
            output = outputs[i,j,:,:]
            fl_output = output.view(-1, 1)
            result = pixel_encoder(fl_output)
            df_result = df_result.append({"Image_Label": f"{img_names[i]}_{label_dict[j]}", "EncodedPixels": result}, ignore_index=True)

df_result.to_csv("understanding_cloud_organization/test_result.csv") 
    
            




    



