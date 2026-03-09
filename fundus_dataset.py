import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image

class FundusDataset(Dataset):
    def __init__(self, df, img_root, high_quality_tf=None, low_quality_tf=None, label_col='final_quality'):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.high_quality_tf = high_quality_tf
        self.low_quality_tf = low_quality_tf
        self.label_col = label_col
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_root, row["file"])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        label = int(row[self.label_col] > 0)

        if label == 0:
            img = self.low_quality_tf(image=img)["image"]
        else:
            img = self.high_quality_tf(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long), row["file"]


