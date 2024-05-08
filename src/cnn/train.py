import os
import torch

def eval_on_test_set(model):
    model.eval()
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model.to(device)
    running_accuracy = 0
    loss=0

    for data in test_loader:
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

# Copy from CA of CEG5304
def train_for_one_epoch(model):
    model.train()
    # Set up device
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    print(f"Using device {device} to train the model.")
    model.to(device)

    # set the running quatities to zero at the beginning of the epoch
    running_loss = 0
    running_accuracy = 0

    for data in train_loader:
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