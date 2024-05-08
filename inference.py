import torch
from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference(model, image):
    model.eval()
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def main():
    model = Model(num_classes=2)
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)

    image =...  # load your image here
    predicted_label = inference(model, image)
    print(f'Predicted label: {predicted_label}')

if __name__ == '__main__':
    main()
