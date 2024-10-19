import pandas as pd
import requests
import torch
from io import BytesIO
from PIL import Image
import pydicom
from torch.utils.data import Dataset, DataLoader
from models import BaselineModel, EfficientNetEncoder
from torchvision import transforms

class TestReflaxcDataset(Dataset):
    def __init__(self, csv, transform=None, num_images=10, diseases=[]):
        self.transform = transform
        self.data = []
        self.labels = []
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            if num_images and len(self.data) > num_images:
                break
            self.data.append(row['image'])
            self.labels.append(row[diseases].values)

    def __len__(self):
        return len(self.data)  

    def __getitem__(self, idx):
        image_url = "https://" + self.data[idx]
        try:
            response = requests.get(image_url, timeout=5) 
            response.raise_for_status()
            image = pydicom.dcmread(BytesIO(response.content)).pixel_array
            if self.transform:
                image = self.transform(image)  
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image {image_url}: {e}")
            image = None  
        return image, self.labels[idx] if image else None


data_csv = "Dataset\\reflacx-reports-and-eye-tracking-data-for-localization-of-abnormalities-in-chest-x-rays-1.0.0\main_data\metadata_phase_1.csv"
sample_diseases = [
    "Airway wall thickening",
    "Atelectasis",
    "Consolidation",
    "Emphysema",
    "Enlarged cardiac silhouette",
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = TestReflaxcDataset(csv=data_csv, diseases=sample_diseases, transform=data_transform)
test_dataloader = DataLoader(dataset=dataset)
model = BaselineModel()

test_image = torch.randn((1, 3, 224, 224))  
output = model(test_image)
print(output)

# with torch.no_grad():
#     model.eval()
#     for input, label in test_dataloader:
#         input = input.to(device)    
#         label = input.to(device)
        
#         output = model(input)
