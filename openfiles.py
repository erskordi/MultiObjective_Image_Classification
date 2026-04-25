from pathlib import Path
import h5py
import pandas as pd
import os

if __name__ == "__main__":
    data_name_conversion_dict ={
            
        }
    
    data_dir = Path("/home/erskordi/Documents/UNM-files/Spring26/ECE533/Project/Data")
    hdf5_paths = list(data_dir.glob("5x64x64*.hdf5"))
    
    train_dataloader, valid_dataloader, test_dataloader = image_data_import()