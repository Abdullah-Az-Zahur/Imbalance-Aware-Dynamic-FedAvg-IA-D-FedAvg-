import os
import shutil
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import kagglehub

def setup_ham10000_dataset(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "HAM10000_images")
    metadata_csv = os.path.join(data_dir, "HAM10000_metadata.csv")
    
    # Check 1: Extract/Setup Already Done?
    if os.path.exists(images_dir) and os.path.exists(metadata_csv):
        if len(os.listdir(images_dir)) > 0:
            print("[INFO] Dataset already set up. Skipping download.")
            return images_dir, metadata_csv

    print("[INFO] Downloading HAM10000 dataset via KaggleHub...")
    try:
        # Download latest version via KaggleHub
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print(f"[SUCCESS] Downloaded to cache path: {path}")

        # Organize images into ./data/HAM10000_images
        os.makedirs(images_dir, exist_ok=True)
        
        print("[INFO] Organizing dataset files...")
        # HAM10000 split images in two folders (part_1 and part_2)
        for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            part_path = os.path.join(path, part)
            if os.path.exists(part_path):
                files = os.listdir(part_path)
                for f in tqdm(files, desc=f"Moving {part}"):
                    src = os.path.join(part_path, f)
                    dst = os.path.join(images_dir, f)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)

        # Copy Metadata CSV
        src_csv = os.path.join(path, "HAM10000_metadata.csv")
        if os.path.exists(src_csv):
            shutil.copy(src_csv, metadata_csv)

        print("[SUCCESS] HAM10000 Dataset ready for training!")
        return images_dir, metadata_csv

    except Exception as e:
        print(f"[ERROR] Failed to download automatically: {e}")
        print("[TIP] You can manually download HAM10000 from Kaggle and put images in './data/HAM10000_images' and CSV at './data/HAM10000_metadata.csv'")
        return None, None


# --- PyTorch Custom Dataset Class ---
class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve image ID and target label
        img_id = self.df.iloc[idx]['image_id']
        label = self.df.iloc[idx]['cell_type_idx']
        
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    setup_ham10000_dataset()