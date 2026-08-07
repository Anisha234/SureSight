import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MultiImageFundusDataset(Dataset):
    """
    Groups all images for a patient into a single sample.

    Returns (imgs_tensor, mask, label, patient_id) where:
      - imgs_tensor: (num_images, C, H, W) — zero for padded slots
      - mask:        (num_images,) bool — True marks slots to be ignored by the transformer
      - label:       1 if any image shows disease, else 0
      - patient_id:  patient identifier

    img_indices: optional list of integer indices into the patient's image list.
        E.g. img_indices=[0, 2] loads the 1st and 3rd images and places them in
        slots 0 and 1; slots 2+ are zero-padded and marked True in the mask.
        None (default) loads all available images up to num_images.
    """

    def __init__(self, df, img_root, transform=None, label_col='final_icdr',
                 patient_col='patient', num_images=4, img_indices=None):
        self.img_root    = img_root
        self.transform   = transform
        self.num_images  = num_images
        self.img_indices = img_indices

        self.patients = []
        for patient_id, group in df.groupby(patient_col):
            files = group['file'].tolist()
            label = int((group[label_col] > 0).any())
            self.patients.append((patient_id, files, label))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id, files, label = self.patients[idx]

        # Select which files to load
        if self.img_indices is not None:
            selected = [files[i] for i in self.img_indices if i < len(files)]
        else:
            selected = files[:self.num_images]

        images = []
        for f in selected:
            img = np.array(Image.open(os.path.join(self.img_root, f)).convert("RGB"))
            if self.transform:
                img = self.transform(image=img)["image"]
            images.append(img)

        n_real = len(images)

        # Zero-pad remaining slots
        if images:
            pad = torch.zeros_like(images[0])
        else:
            pad = torch.zeros(3, 392, 392)

        while len(images) < self.num_images:
            images.append(pad)

        imgs_tensor = torch.stack(images)  # (num_images, C, H, W)

        # True = padded slot, to be masked out in the transformer
        mask = torch.zeros(self.num_images, dtype=torch.bool)
        mask[n_real:] = True

        return imgs_tensor, mask, torch.tensor(label, dtype=torch.long), patient_id
