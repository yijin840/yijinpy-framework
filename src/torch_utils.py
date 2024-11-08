import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class TorchUtils:
    def __init__(self):
        # Download training data from open datasets.
        self.training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )
        # Download test data from open datasets.
        self.test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
        self.batch_size = 64
        # Create data loaders.
        self.train_dataloader = DataLoader(
            self.training_data, batch_size=self.batch_size
        )
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)

    def printData(self):
        print("print data:")
        for X, y in self.test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
