import torch
import torch.nn as nn
import os
from utils.plot import plot_metric
from utils.optimizer import get_optimizer
from utils.train_utils import (train_one_epoch, evaluate)
from models.mlp import BinaryClassifier
from utils.dataset import get_dataloader

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

EPOCHS = 10


def run_experiment(
        optimizer_name,
        learning_rate
    ):

    train_loader, test_loader = get_dataloader()

    model = BinaryClassifier().to(DEVICE)

    criterion = nn.BCELoss()

    optimizer = get_optimizer(
        optimizer_name,
        model.parameters(),
        learning_rate
    )

    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(
            f"{optimizer_name.upper()} "
            f"Epoch {epoch + 1}/{EPOCHS}"
            f"| Train Loss : {train_loss :.4f}"
            f"| Train Accuracy : {train_accuracy :.4f}"
            f"| Test Loss : {test_loss : .4f}"
            f"| Test Accuracy : {test_accuracy : .4f}"
        )

    os.makedirs("checkpoints", exist_ok = True)
    
    
    torch.save(
        model.state_dict(),
        "checkpoints/mnist_01.pth"
    )


    return{
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies
    }


# sgd_result = run_experiment(
#     optimizer_name = "sgd",
#     learning_rate = 0.01
# )

# momentum_result = run_experiment(
#     optimizer_name = "momentum",
#     learning_rate = 0.01,
# )

adam_result = run_experiment(
    optimizer_name = "adam",
    learning_rate = 0.001
)


# plot_metric(sgd_result["train_losses"],
#             momentum_result["train_losses"],
#             adam_result["train_losses"],
#             title = "Training loss comparison",
#             y_label = "Loss")

# plot_metric(sgd_result["test_losses"],
#             momentum_result["test_losses"],
#             adam_result["test_losses"],
#             title = "Test loss comparison",
#             y_label = "Loss")


# plot_metric(sgd_result["train_accuracies"],
#             momentum_result["train_accuracies"],
#             adam_result["train_accuracies"],
#             title = "Training accuracy comparison",
#             y_label = "Accuracy")


# plot_metric(sgd_result["test_accuracies"],
#             momentum_result["test_accuracies"],
#             adam_result["test_accuracies"],
#             titel = "Test accuracy comparison",
#             y_label = "Accuracy")

