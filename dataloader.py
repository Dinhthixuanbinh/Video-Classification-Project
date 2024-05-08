from  torchvision import transforms 
from VideoDataset import VideoDataset
from torch.utils.data import DataLoader

import os
BATCH_SIZE = 16
MAX_LEN = 15
IMAGE_SIZE = 224

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

# Load dataset
train_dataset = VideoDataset(
    root_dir= "./rwf-2000", phase = "train", 
    transform=transform , n_frames= MAX_LEN
)

val_dataset = VideoDataset(
    root_dir= "./rwf-2000", phase = "val", 
    transform=transform , n_frames= MAX_LEN
)
# Count number of cpus
cpus = os.cpu_count()
print(f"Number of cpus: {cpus}")

 # Create data loaders
train_loader = DataLoader (
  train_dataset, batch_size= BATCH_SIZE, num_workers= cpus, shuffle= True
 )

val_loader = DataLoader (
  val_dataset, batch_size= BATCH_SIZE, num_workers= cpus, shuffle= False
 )