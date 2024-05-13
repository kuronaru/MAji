import os
import time
from logging import INFO, DEBUG

import torch
from matplotlib import pyplot as plt
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cnn.dataset import TileDataset
from src.cnn.tile_classifier import TileClassifier
from src.utils.dbgf import DebugPrintf

dbgf = DebugPrintf("tile_classifier_train", INFO)

LOAD_MODEL = False
SAVE_MODEL = True
dbgf(INFO, "Load model = %s, save model = %s" % (LOAD_MODEL, SAVE_MODEL))
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
dbgf(INFO, "Using device %s" % DEVICE)

# Training parameters
epochs = 100
batch_size = 8
learning_rate = 0.1


def train_model(model, data_loader, device=DEVICE):
    model.train()
    model.to(device)

    # set the running quantities to zero at the beginning of the epoch
    running_loss = 0
    running_accuracy = 0

    for data in data_loader:
        inputs, labels = data
        labels = torch.flatten(labels).type(torch.LongTensor)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += (outputs.argmax(1) == labels).sum().cpu()

    total_loss = running_loss / train_size
    total_accuracy = running_accuracy / train_size
    return total_loss, total_accuracy


def evaluate_model(model, data_loader, device=DEVICE):
    model.eval()
    model.to(device)
    running_accuracy = 0
    loss = 0

    for data in data_loader:
        inputs, labels = data
        labels = torch.flatten(labels).type(torch.LongTensor)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss += loss_func(outputs, labels).item()
        running_accuracy += (outputs.argmax(1) == labels).sum().cpu()

    total_loss = loss / test_size
    total_accuracy = running_accuracy / test_size
    dbgf(DEBUG, "Evaluation: loss=%.4f \t accuracy=%.4f" % (total_loss, total_accuracy * 100))
    return total_loss, total_accuracy


# Build dataset
# Calculate mean and std
# mean: tensor([0.6710, 0.6679, 0.6726])
# std: tensor([0.2327, 0.2291, 0.2132])

# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = TileDataset("../../data/tiles", 8, transform)
# dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# mean = torch.zeros(3)
# std = torch.zeros(3)
# for img, _ in dataloader:
#     img = img.view(img.size(0), img.size(1), -1)
#     mean += img.mean(2).sum(0)
#     std += img.std(2).sum(0)
# mean /= len(dataloader)
# std /= len(dataloader)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.6710, 0.6679, 0.6726), (0.2327, 0.2291, 0.2132))])

train_dataset = TileDataset("../../data/tiles", 8, transform=transform)
test_dataset = TileDataset("../../data/tiles", 1, 5, transform=transform)
train_size = len(train_dataset)
test_size = len(test_dataset)
dbgf(INFO, "Train dataset size %d, test dataset size %d" % (train_size, test_size))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# data_loaders = {"train": train_loader, "test": test_loader}
# dataset_sizes = {"train": train_size, "test": test_size}

# Load the trained model
trained_model = TileClassifier()
if (LOAD_MODEL):
    assert os.path.exists("../../data/model/model_tile_classifier.pt"), "model_tile_classifier does not exist"
    trained_model.load_state_dict(torch.load("../../data/model/model_tile_classifier.pt",
                                             map_location=torch.device("cpu")))

# Optimizer and loss function
optimizer = optim.SGD(trained_model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

metrics = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
for epoch in tqdm(range(epochs), total=epochs, desc="Training"):
    start = time.time()
    train_loss_epoch, train_acc_epoch = train_model(trained_model, train_loader)
    test_loss_epoch, test_acc_epoch = evaluate_model(trained_model, test_loader)
    metrics["train_loss"].append(train_loss_epoch)
    metrics["train_acc"].append(train_acc_epoch)
    metrics["test_loss"].append(test_loss_epoch)
    metrics["test_acc"].append(test_acc_epoch)

# Plot result figure
plt.subplot(1, 2, 1)
plt.title("Tile Classifier")
plt.plot(range(epochs), metrics["train_loss"], range(epochs), metrics["test_loss"], marker=".")
plt.legend(labels=["train loss", "test loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(range(epochs), metrics["train_acc"], range(epochs), metrics["test_acc"], marker="x")
plt.legend(labels=["train accuracy", "test accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Save trained model
if (SAVE_MODEL):
    save_dir = "../../data/model"
    if (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    f = os.path.join(save_dir, "model_tile_classifier.pt")
    torch.save(trained_model.state_dict(), f)
