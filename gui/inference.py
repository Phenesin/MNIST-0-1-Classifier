import torch

from models.mlp import BinaryClassifier

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = BinaryClassifier().to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/mnist_01.pth",
        map_location = DEVICE 
    )
)
model.eval()


def predict(tensor):
    with torch.no_grad():
        output = model(tensor)
        probability_0 = output.item()
        probability_1 = 1 - probability_0
    return probability_0, probability_1