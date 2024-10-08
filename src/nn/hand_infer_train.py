import os
from logging import INFO, DEBUG

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.nn.dataset import MajDataset
from src.nn.hand_infer import HandInfer
from src.utils.data_process import code_to_data, data_to_code
from src.utils.dbgf import DebugPrintf, HAND_INFER_DBG_LVL

dbgf = DebugPrintf("hand_infer_train", HAND_INFER_DBG_LVL)

LOAD_MODEL = False
SAVE_MODEL = True
dbgf(INFO, "Load model = %s, save model = %s" % (LOAD_MODEL, SAVE_MODEL))
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
dbgf(INFO, "Using device %s" % DEVICE)

# training parameters
one_hot_size = 38
embed_size = 50
hidden_size = 128
num_layers = 1
learning_rate = 0.01
batch_size = 1
epochs = 10


def train_model(model, optimizer, loss_func, data_loader, train_size, device=DEVICE):
    model.train()
    model.to(device)

    running_loss = 0
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=13).to(device)

    for data in data_loader:
        inputs, targets = data
        inputs = inputs.squeeze(0)
        targets = targets.squeeze(0)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, targets)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        accuracy(outputs, targets)
        running_loss += loss.item()

    total_loss = running_loss / train_size
    total_accuracy = accuracy.compute().cpu()
    return total_loss, total_accuracy


def evaluate_model(model, loss_func, data_loader, test_size, device=DEVICE):
    model.eval()
    model.to(device)

    running_loss = 0
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=13).to(device)

    for data in data_loader:
        inputs, targets = data
        inputs = inputs.squeeze(0)
        targets = targets.squeeze(0)
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)  # output size batch_size * 190
            for code in outputs:
                predicted_hand = code_to_data(code)
                dbgf(DEBUG, "Predicted hand: %s" % predicted_hand)

        accuracy(outputs, targets)
        running_loss += loss_func(outputs, targets).item()

    total_loss = running_loss / test_size
    total_accuracy = accuracy.compute().cpu()
    dbgf(DEBUG, "Evaluation: loss=%.4f \t accuracy=%.4f" % (total_loss, total_accuracy))
    return total_loss, total_accuracy


def run():
    trained_model = "../../data/model/model_hand_infer.pt"
    model = HandInfer(one_hot_size, embed_size, hidden_size, num_layers)
    if (LOAD_MODEL):
        assert os.path.exists(trained_model), "model_hand_infer does not exist"
        model.load_state_dict(torch.load(trained_model, map_location=torch.device("cpu")))

    train_dataset = MajDataset("../../data/games", 60, 0)
    test_dataset = MajDataset("../../data/games", 3, 60)
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    dbgf(INFO, "Train dataset size %d, test dataset size %d" % (train_size, test_size))

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    metrics = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in tqdm(range(epochs), total=epochs, desc="Training"):
        train_loss_epoch, train_acc_epoch = train_model(model, optimizer, loss_func, train_loader, train_size)
        test_loss_epoch, test_acc_epoch = evaluate_model(model, loss_func, test_loader, test_size)
        metrics["train_loss"].append(train_loss_epoch)
        metrics["train_acc"].append(train_acc_epoch)
        metrics["test_loss"].append(test_loss_epoch)
        metrics["test_acc"].append(test_acc_epoch)

    # plot result figure
    plt.subplot(1, 2, 1)
    plt.title("Hand Infer")
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

    # save trained model
    if (SAVE_MODEL):
        save_dir = "../../data/model"
        if (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        f = os.path.join(save_dir, "model_hand_infer.pt")
        torch.save(model.state_dict(), f)


if __name__ == "__main__":
    run()
