import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image

class FundusDataset(Dataset):
    def __init__(self, df, img_root, high_quality_tf=None, low_quality_tf=None,
                 label_col='final_quality', img_idx=None):
        self.img_root = img_root
        self.high_quality_tf = high_quality_tf
        self.low_quality_tf = low_quality_tf
        self.label_col = label_col

        # Always use patient-level label: 1 if any image for that patient shows disease
        pat_label = (
            df.groupby('patient')[label_col]
            .apply(lambda x: int((x > 0).any()))
            .rename('_pat_label')
        )
        df = df.merge(pat_label, on='patient')
        df[label_col] = df['_pat_label']
        df = df.drop(columns=['_pat_label'])

        if img_idx is not None:
            # Select the img_idx-th row per patient; fall back to last if fewer images
            df = (
                df.groupby('patient', group_keys=False)
                .apply(lambda g: g.iloc[[min(img_idx, len(g) - 1)]])
                .reset_index(drop=True)
            )

        self.df = df.reset_index(drop=True)
        
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


