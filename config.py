from torchvision import transforms
import torch
# Hyperparameters
NUM_CLASSES = 2
NUM_FRAMES = 15
BATCH_SIZE = 2
MAX_LEN = 15
IMAGE_SIZE = 224
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5

# Dataset and data loader settings
ROOT_DIR = "./dataset/rwf-2000"
PHASES = ["train", "val"]
TRANSFORMS = [
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
]

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
