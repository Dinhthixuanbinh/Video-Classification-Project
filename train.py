import time
import torch
import logging
from tqdm import tqdm
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls")
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def colorstr(*input):
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]

def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device="cuda"
):
    """
    Function to train the model.

    Parameters:
    - model: The neural network model to train.
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    - criterion: The loss function.
    - optimizer: The optimization algorithm.
    - num_epochs: Number of epochs to train for.
    - device: The device to run the training on, 'cuda' or 'cpu'.

    Returns:
    - model: The trained model.
    """
    since = time.time()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_val_acc = 0.0

    # Send the model to the specified device
    model.to(device)

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        LOGGER.info(colorstr(f"Epoch {epoch}/{num_epochs-1}:"))

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.info(
                    colorstr("bright_yellow", "bold", "\n%20s" + "%15s" * 3)
                    % ("Training:", "gpu_mem", "loss", "acc")
                )
                model.train()
            else:
                LOGGER.info(
                    colorstr("bright_green", "bold", "\n%20s" + "%15s" * 3)
                    % ("Validation:", "gpu_mem", "loss", "acc")
                )
                model.eval()

            running_items = 0
            running_loss = 0.0
            running_corrects = 0

            # Use the appropriate data loader
            data_loader = train_loader if phase == "train" else val_loader

            _phase = tqdm(
                data_loader,
                total=len(data_loader),
                bar_format="{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}",
                unit="batch",
            )

            # Iterate over data.
            for inputs, labels in _phase:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_items += outputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / running_items
                epoch_acc = running_corrects / running_items

                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB"
                desc = ("%35s" + "%15.6g" * 2) % (
                    mem,
                    epoch_loss,
                    epoch_acc,
                )
                _phase.set_description_str(desc)

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    history["best_epoch"] = epoch

                print(f"Best val Acc: {best_val_acc:4f}")

    time_elapsed = time.time() - since
    history["INFO"] = (
        "Training complete in {:.0f}h {:.0f}m {:.0f}s with {} epochs - Best val Acc: {:4f}".format(
            time_elapsed // 3600,
            time_elapsed % 3600 // 60,
            time_elapsed % 60,
            num_epochs,
            best_val_acc,
        )
    )

    return model