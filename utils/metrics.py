import torch


def binary_accuracy(outputs, labels):
    predictions = (outputs >= 0.5).float()
    correct = (predictions == labels).sum().item()
    accuracy = correct/labels.size(0)
    return accuracy