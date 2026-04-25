import h5py
import pandas as pd
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np

class HDF5ImageDataset(Dataset):
    def __init__(self, h5_path, csv_path, dataset_name="image", transform=None):
        self.h5_path = h5_path
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.transform = transform
        self._file = None
        self._images = None

        # Read length once without keeping file open
        with h5py.File(self.h5_path, "r") as f:
            #print(f"Available datasets in {self.h5_path}: {list(f.keys())}")
            self.length = f[self.dataset_name].shape[0]

    def _lazy_init(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
            self._images = self._file[self.dataset_name]
            self._csv = pd.read_csv(self.csv_path)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._lazy_init()
        img = self._images[idx]

        img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)

        specz_redshift = self._csv.iloc[idx]["specz_redshift"]
        label = self._csv.iloc[idx]["outlier_label"]
        specz_redshift = torch.tensor(specz_redshift, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return img, specz_redshift, label

def image_data_import():
    def csv_data_import():
        data_dir = Path("/home/erskordi/Documents/UNM-files/Spring26/ECE533/Project/Data")
        csv_paths = list(data_dir.glob("galaxy_labels_redshift_outlier_*.csv"))
        for path in csv_paths:
            if "training" in str(path):
                training_parts = path.parts
            elif "validation" in str(path):
                validation_parts = path.parts
            elif "test" in str(path):
                test_parts = path.parts
        training_csv_path = os.path.join(*training_parts)
        validation_csv_path = os.path.join(*validation_parts)
        test_csv_path = os.path.join(*test_parts)

        return training_csv_path, validation_csv_path, test_csv_path 
    
    data_dir = Path("/home/erskordi/Documents/UNM-files/Spring26/ECE533/Project/Data")
    hdf5_paths = list(data_dir.glob("5x64x64*.hdf5"))
    
    for path in hdf5_paths:
        if "training" in str(path):
            training_parts = path.parts
        elif "validation" in str(path):
            validation_parts = path.parts
        elif "testing" in str(path):
            test_parts = path.parts
    
    training_hdf5_path = os.path.join(*training_parts)
    validation_hdf5_path = os.path.join(*validation_parts)
    test_hdf5_path = os.path.join(*test_parts)

    training_csv_path, validation_csv_path, test_csv_path = csv_data_import()

    train_dataloader = DataLoader(
        HDF5ImageDataset(h5_path=training_hdf5_path, csv_path=training_csv_path), 
        batch_size=64, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True)

    valid_dataloader = DataLoader(
        HDF5ImageDataset(h5_path=validation_hdf5_path, csv_path=validation_csv_path), 
        batch_size=64, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_dataloader = DataLoader(
        HDF5ImageDataset(h5_path=test_hdf5_path, csv_path=test_csv_path), 
        batch_size=64, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_dataloader, valid_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, valid_dataloader, test_dataloader = image_data_import()

    for batch in train_dataloader:
        imgs, specz_redshifts, labels = batch
        print(imgs.shape)  # Process your batch here
        for i, img in enumerate(imgs):
            print(img.shape)  # (5, 64, 64)
            # Visualize each of the 5 channels as a subplot
            fig, axes = plt.subplots(1, img.shape[0], figsize=(15, 3))
            print(f"Redshift: {specz_redshifts[i]}, Label: {labels[i]}")
            for ch, ax in enumerate(axes):
                ax.imshow(img[ch].numpy(), cmap="hot")
                ax.set_title(f"Channel {ch}")
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            break
        break  # Remove this break to process the entire dataset