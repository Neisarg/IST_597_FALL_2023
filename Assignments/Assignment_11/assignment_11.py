import torch
import random
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import pprint
from torch.utils.data.dataloader import DataLoader


"""
Convolution Neural Network: Feel free to change the number of layers, 
size of layers and activation functions.

https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

"""
class ConvolutionNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(ConvolutionNeuralNetwork, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(128, 128, [5,5]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, [5,5]),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2)
        )

        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, data):
        bs = data.shape[0]
        data = self.conv(data)
        data = data.reshape(bs, -1)
        data = torch.relu(self.fc1(data))
        data = torch.relu(self.fc2(data))
        logits = self.fc3(data)
        return logits


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
       device = torch.device("cpu")

    seed = 1234  # change this seed to run trials
    random.seed(seed)
    torch.manual_seed(seed)


    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=T.ToTensor()
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=T.ToTensor()
    )

    val_dataset, train_dataset = torch.utils.data.random_split(train_dataset, [10000, 40000])

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    class_to_idx = {
        "airplane"    : 0,
        "automobile"  : 1,
        "bird"        : 2,
        "cat"         : 3,
        "deer"        : 4,
        "dog"         : 5,
        "frog"        : 6,
        "horse"       : 7,
        "ship"        : 8,
        "truck"       : 9
    }
    idx_to_class = {i:c for c, i in class_to_idx.items()}

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[sample_idx]
        label = idx_to_class[label]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0).squeeze())
    plt.show()

    batch_size = 1024

    train_data = DataLoader(train_dataset, batch_size = batch_size)
    test_data = DataLoader(test_dataset, batch_size = batch_size)
    val_data = DataLoader(val_dataset, batch_size = batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    model = ConvolutionNeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0, momentum = 0.9)

    max_epochs = 50

    for epoch in range(max_epochs):
        for i, batch_data in enumerate(train_data):
            # Write your training loop here 