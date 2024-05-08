import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from dataloader import train_loader, val_loader
from loss import MyLoss, evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def fine_tune(model, device, train_loader, optimizer, criterion, num_classes):
    for param in model.parameters():
        param.requires_grad = True
    for i in range(num_classes):
        model.vivit.fc.weight.data[i] = model.vivit.fc.weight.data[i] / num_classes
    model.vivit.fc.bias.data = model.vivit.fc.bias.data / num_classes
    return train(model, device, train_loader, optimizer, criterion)

def main():
    num_classes = 2
    num_epochs = 10
    learning_rate = 0.001

    model = Model(num_classes)
    model.to(device)

    criterion = MyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loss = train(model, device, train_loader, optimizer, criterion)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
        accuracy_val, precision_val, recall_val, f1_score_val = evaluate(model, val_loader)
        print(f'Epoch {epoch+1}, Val Accuracy: {accuracy_val:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1-score: {f1_score_val:.4f}')

    fine_tune(model, device, train_loader, optimizer, criterion, num_classes)

if __name__ == '__main__':
    main()
