from mlp import BinaryClassifier
import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = BinaryClassifier().to(DEVICE)

model.load_state_dict(torch.load("01_classifier/checkpoints/mnist_01.pth"))

for name, param in model.named_parameters():
    print(name, param.shape)