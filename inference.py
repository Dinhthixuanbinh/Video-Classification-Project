import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from config import *

from model import Model
from dataset import VideoDataset
from train import train_model

# Create model
model = Model(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)

# Check param
param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model has {param} parameters")

# Test the model with a random input (batch_size, channels, frames, height, width)
inputs = torch.rand(1, 3, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE)
output = model(inputs)
print(output.shape)

# Load dataset
transform = transforms.Compose(TRANSFORMS)
train_dataset = VideoDataset(root_dir=ROOT_DIR, phase="train", transform=transform, n_frames=MAX_LEN)
val_dataset = VideoDataset(root_dir=ROOT_DIR, phase="val", transform=transform, n_frames=MAX_LEN)

# Create data loaders
cpus = os.cpu_count()
print(f"Number of cpus: {cpus}")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=cpus, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=cpus, shuffle=False)

# Test data loader
for data, label in train_loader:
    print(data.shape, label)
    break

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE)

# # Define a predict function
# def predict(model, data_loader, device):
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for data in data_loader:
#             data = data.to(device)
#             output = model(data)
#             _, predicted = torch.max(output, 1)
#             predictions.extend(predicted.cpu().numpy())
#     return predictions
def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:  # iterate over data and labels, but ignore labels
            inputs = inputs.to(device)
            output = model(inputs)
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

# Make predictions on the validation set
val_predictions = predict(trained_model, val_loader, DEVICE)
print("Validation predictions:", val_predictions)

# def predict(model, data_loader, device, dataset):
#     model.eval()
#     predictions = []
#     labels = []
#     image_names = []
#     with torch.no_grad():
#         for inputs, labels_batch in data_loader:  # iterate over data and labels
#             inputs = inputs.to(device)
#             output = model(inputs)
#             _, predicted = torch.max(output, 1)
#             predictions.extend(predicted.cpu().numpy())
#             labels.extend(labels_batch.cpu().numpy())
#             image_names.extend([dataset.video_names[i] for i in range(labels_batch.shape[0])])  # get the image names from the dataset
#     return predictions, labels, image_names

# # Make predictions on the validation set
# val_predictions, val_labels, val_image_names = predict(trained_model, val_loader, DEVICE, val_dataset)
# print("Validation predictions:")
# for i, (prediction, label, image_name) in enumerate(zip(val_predictions, val_labels, val_image_names)):
#     print(f"Image: {image_name}, Prediction: {prediction}, Label: {label}")
#     plt.imshow(val_dataset[i][0].permute(1, 2, 0))  # display the image
#     plt.title(f"Prediction: {prediction}, Label: {label}")
#     plt.show()