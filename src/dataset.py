import os
import zipfile
import urllib.request
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Automatic Download & Extract Helper ---
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def setup_ham10000_dataset(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    extract_path = os.path.join(data_dir, "HAM10000_images")
    zip_path = os.path.join(data_dir, "skin-cancer-mnist-ham10000.zip")
    
    # Check 1: Extraction Complete?
    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        print("[INFO] Extracted dataset already exists. Skipping download and extraction.")
        return extract_path

    # Check 2: Zip File Exists? If not, Download
    if not os.path.exists(zip_path):
        print("[INFO] Downloading HAM10000 Dataset...")
        # Direct public mirror URL for HAM10000 (Kaggle direct mirror)
        url = "https://dataverse.harvard.edu/api/access/datafile/3037206" # Metadata & images archive
        
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="HAM10000 Download") as t:
                urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            print("[TIP] You can manually place 'HAM10000_images' folder inside the './data' directory.")
            return None
    else:
        print("[INFO] Zip file already downloaded. Skipping download.")

    # Check 3: Extraction Step with Progress Bar
    print("[INFO] Extracting dataset...")
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc="Extracting Files"):
            zip_ref.extract(member, extract_path)
            
    print("[SUCCESS] Dataset setup complete!")
    return extract_path


# --- PyTorch Custom Dataset ---
class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = f"{self.df.iloc[idx, 0]}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # Test script locally
    setup_ham10000_dataset()