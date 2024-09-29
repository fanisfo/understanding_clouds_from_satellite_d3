import os
import numpy as np
import pandas as pd

import torch
from PIL import Image

class LoadData():
    def __init__(
            self, 
            label_dict: dict,
            transformer = None,
            images_dir : str = 'understanding_cloud_organization/train_images', 
            mask_csv_path : str = "understanding_cloud_organization/train.csv"
            ):
        self.label_dict = label_dict
        self.images_dir = images_dir
        self.images = os.listdir(images_dir)
        self.transformer = transformer
        self.mask_csv_path = mask_csv_path
        self.mask_df = self.getmask()
    
    def getmask(self) -> pd.DataFrame:
        mask_df = pd.read_csv(self.mask_csv_path)
        mask_df["Img_Namejpg"] = mask_df["Image_Label"].apply(lambda x: x.split("_")[0])
        mask_df["Label"] = mask_df["Image_Label"].apply(lambda x: x.split("_")[1])
        return mask_df
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Generate corresponding mask as np.array
        img_namejpg = img_path.split("/")[-1]
        mask_img = self.mask_df[self.mask_df["Img_Namejpg"] == img_namejpg]
        mask_np = np.zeros((width, height))
        for label in mask_img["Label"].to_list():
            px = mask_img.loc[mask_img["Label"] == label, "EncodedPixels"].str.split(" ")
            starting_px = px[1::2]
            run_length = px[::2]
            for i in range(len(starting_px)):
                row_index = starting_px[i] // height   
                col_index = starting_px[i] % height 
                mask_np[row_index:row_index + run_length[i], col_index] = self.label_dict[label]

        if self.transformer:
            image = self.transformer(image)
            mask = self.transformer(mask_np)

        return image, mask