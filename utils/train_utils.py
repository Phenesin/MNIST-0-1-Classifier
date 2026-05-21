import torch

from .metrics import binary_accuracy

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device
):
    model.train()

    total_loss = 0
    total_accuracy = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += binary_accuracy(outputs, labels)

    avg_loss = total_loss / len(loader)
    avg_accuracy = total_accuracy / len(loader)

    return avg_loss, avg_accuracy



def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_accuracy += binary_accuracy(outputs, labels)
    avg_loss = total_loss/len(loader)
    avg_accuracy = total_accuracy/len(loader)
    return avg_loss, avg_accuracy