import os
import time

import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

from src.cnn.dataset import MajDataset
from src.cnn.resnet import ResNet18

assert os.path.exists("../../data/model/model_resnet18.pt"), "model_resnet18 does not exist"
LOAD_MODEL = os.path.exists("../../data/model/model_resnet18.pt")
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
print("Using device %s" % DEVICE)


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
    print('Evaluation on test set: loss= {:.3f} \t accuracy= {:.2f}%'.format(total_loss, total_accuracy * 100))

    return total_loss, total_accuracy


# Preprocess dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalization

train_dataset = MajDataset("../../data/image", 12, 0, transform)
test_dataset = MajDataset("../../data/image", 50, 450, transform)

# train_set = [train_dataset]
# train_set = ConcatDataset(train_set)
train_size = len(train_dataset)
# test_set = [test_dataset]
# test_set = ConcatDataset(test_set)
test_size = len(test_dataset)
print("train dataset size %d, test dataset size %d" % (train_size, test_size))

# Build data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
data_loaders = {"train": train_loader, "test": test_loader}
dataset_sizes = {"train": train_size, "test": test_size}

# Load the trained model
trained_model = ResNet18()
if (LOAD_MODEL):
    trained_model.load_state_dict(torch.load("../../data/model/model_resnet18.pt", map_location=torch.device('cpu')))

# Hyper parameters
epochs = 10
batch_size = 256
learning_rate = 0.1

# Set up optimizer and loss function
optimizer = optim.SGD(trained_model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

metrics = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
for epoch in range(epochs):
    start = time.time()
    train_loss_epoch, train_acc_epoch = train_model(trained_model, train_loader)
    elapsed = (time.time() - start) / 60

    test_loss_epoch, test_acc_epoch = evaluate_model(trained_model, test_loader)
    metrics["train_loss"].append(train_loss_epoch)
    metrics["train_acc"].append(train_acc_epoch)
    metrics["test_loss"].append(test_loss_epoch)
    metrics["test_acc"].append(test_acc_epoch)

# Save trained model
torch.save(trained_model.state_dict(), "../../data/model/model_resnet18.pt")
