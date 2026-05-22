import torch

from models.mlp import BinaryClassifier

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = BinaryClassifier().to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/adam.pth",
        map_location = DEVICE 
    )
)
model.eval()


def predict(tensor):
    with torch.no_grad():
        output = model(tensor)
        probability_1 = output.item()
        probability_0 = 1 - probability_1
    return probability_0, probability_1