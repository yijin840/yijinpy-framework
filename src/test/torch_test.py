import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from device import get_device

class TorchTest:
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

    def print_data(self):
        print("print data:")
        for X, y in self.test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def to_device(self):
        return self.to(get_device())


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(get_device()), y.to(get_device())

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(get_device()), y.to(get_device())
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def run():
    device = get_device()
    print(f"Using {device} device")

    print("start run torch_utils run method.")
    tu = TorchTest()
    tu.print_data()
    model = NeuralNetwork().to_device()
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    print("loss_fn: ", loss_fn)
    print("optimizer: ", optimizer)

    print("run method end.")
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(tu.train_dataloader, model, loss_fn, optimizer)
        test(tu.test_dataloader, model, loss_fn)
    print("Done!")

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")


def load_model_and_eval(model_path, a, b, c, d):
    model = NeuralNetwork().to(get_device())
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 模型预测
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    model.eval()
    tu = TorchTest()
    tu.print_data()
    x, y = tu.test_data[a][b], tu.test_data[c][d]
    with torch.no_grad():
        x = x.to(get_device())
        print("x: ", x)
        pred = model(x)
        print("pred: ", pred)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
