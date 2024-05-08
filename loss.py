import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(output, target)

def accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct / len(target)

def precision(output, target):
    _, predicted = torch.max(output, 1)
    true_positives = (predicted == target).sum().item()
    false_positives = (predicted!= target).sum().item()
    return true_positives / (true_positives + false_positives)

def recall(output, target):
    _, predicted = torch.max(output, 1)
    true_positives = (predicted == target).sum().item()
    false_negatives = (predicted!= target).sum().item()
    return true_positives / (true_positives + false_negatives)

def f1_score(output, target):
    precision_val = precision(output, target)
    recall_val = recall(output, target)
    return 2 * (precision_val * recall_val) / (precision_val + recall_val)

def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += len(labels)
    accuracy_val = total_correct / total_samples
    return accuracy_val, precision(outputs, labels), recall(outputs, labels), f1_score(outputs, labels)
