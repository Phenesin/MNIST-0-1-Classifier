from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size = 64):
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root = "./data",
        train = True,
        download = True,
        transform = transform
    )

    test_dataset = datasets.MNIST(
        root = "./data",
        train = False,
        download = True,
        transform = transform
    )

    train_mask = (
        (train_dataset.targets == 0) |
        (train_dataset.targets == 1)
    )

    test_mask = (
        (test_dataset.targets == 0)|
        (test_dataset.targets == 1)
    )

    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = test_dataset.targets[test_mask]
    
    
    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = train_dataset.targets[train_mask]


    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size = batch_size
    )

    return train_loader, test_loader