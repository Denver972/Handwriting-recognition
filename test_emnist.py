# Training a model on EMNIST dataset
import os
# import cv2 as cv
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torchvision.datasets import EMNIST
# from skimage import io, transform
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import matplotlib.pyplot as plt
# from PIL import Image
import numpy as np
import copy
from tqdm import tqdm
# import pandas as pd

# Choose divice to run the program on. Default to CUDA because it is much
# faster than the cpu
if torch.cuda.is_available():
    device = torch.device('cuda')  # Default CUDA device

else:
    device = torch.device('cpu')
print("Cuda version:", torch.version.cuda)
print("Cuda device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))
print(device)

# n_epochs = 800
# batch_size_train = 100
# batch_size_test = 100
# learning_rate = 0.01
# momentum = 0.5
# log_interval = 10
# loss_NLL = torch.nn.NLLLoss()
# loss_MSE = torch.nn.MSELoss()
# loss_CTC = torch.nn.CTCLoss(10, 10)

root_dir = "EMNIST_Folder"


# transforms = torchvision.transforms.Compose([
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(
#                 (0.1307,), (0.3081,))
#         ])

# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.EMNIST(
#         root=root_dir, split="balanced", train=True, download=True,
#         transform=torchvision.transforms.Compose([
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(
#                 (0.1307,), (0.3081,))
#         ])),
#     batch_size=batch_size_train, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.EMNIST(
#         root_dir, split="balanced", train=False, download=True,
#         transform=torchvision.transforms.Compose([
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(
#                 (0.1307,), (0.3081,))
#         ])),
#     batch_size=batch_size_test, shuffle=True)

full_data = torchvision.datasets.EMNIST(
    root="emnistFolder", split="balanced", download=True)

print(f"All classes: {full_data.classes}")
print(f"Data size: {full_data.data.shape}")

# 112800 for 47 classes, 124800 for 26 classes
images = full_data.data.view([112800, 1, 28, 28]).float()
print(f"Tensor shape: {images.shape}")

# normalise between 0 and 1
images /= torch.max(images)

# visualise the images
character_categories = full_data.classes[1:]
labels = copy.deepcopy(full_data.targets)  #- 1
print(f"Labels: {labels}")
print(f"minimum: {labels.min()}, maximum:{labels.max()}")
print(f"Lable shape: {labels.shape}")
print(torch.sum(labels == 0))


fig, axes = plt.subplots(3, 7, figsize=(13, 6))

for i, ax in enumerate(axes.flatten()):
    which_pic = np.random.randint(images.shape[0])

    image = images[which_pic, 0, :, :].detach()
    letter = character_categories[labels[which_pic]-1]# add -1 inside when 47 classes

    ax.imshow(image.T, cmap='gray')
    ax.set_title(f"Letter: {letter}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# Training test split
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.01)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
batch_size = 128
train_dl = DataLoader(train_data, batch_size=batch_size,
                      shuffle=True, drop_last=True)
test_dl = DataLoader(test_data, batch_size=len(test_data))

# make the model


def make_the_model(print_toggle):
    class EMNISTNet(nn.Module):
        def __init__(self, print_toggle):
            super().__init__()
            self.print_toggle = print_toggle

            # Conv1
            self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, padding=1)
            self.bnorm1 = nn.BatchNorm2d(num_features=64)

            # Conv2
            self.conv2 = nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3)
            # Input: number of channels
            self.bnorm2 = nn.BatchNorm2d(num_features=128)

            # Conv3
            self.conv3 = nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3)
            # Input: number of channels
            self.bnorm3 = nn.BatchNorm2d(num_features=256)

            self.fc1 = nn.Linear(in_features=2*2*256, out_features=256)
            self.fc2 = nn.Linear(in_features=256, out_features=64)
            self.fc3 = nn.Linear(in_features=64, out_features=47) #change here to 47

        def forward(self, x):
            if self.print_toggle:
                print(f"Input: {list(x.shape)}")

            # First Block: conv -> max_pool -> bnorm -> relu
            x = F.max_pool2d(self.conv1(x), 2)
            x = F.leaky_relu((self.bnorm1(x)))
            x = F.dropout(x, p=0.25, training=self.training)
            if self.print_toggle:
                print(f"First Block: {list(x.shape)}")

            # Second Block: conv -> max_pool -> bnorm -> relu
            x = F.max_pool2d(self.conv2(x), 2)
            x = F.leaky_relu((self.bnorm2(x)))
            x = F.dropout(x, p=0.25, training=self.training)
            if self.print_toggle:
                print(f"Second Block: {list(x.shape)}")

            # Third Block: conv -> max_pool -> bnorm -> relu
            x = F.max_pool2d(self.conv3(x), 2)
            x = F.leaky_relu((self.bnorm3(x)))
            x = F.dropout(x, p=0.25, training=self.training)
            if self.print_toggle:
                print(f"Second Block: {list(x.shape)}")

            # Reshape for linear layer
            n_units = x.shape.numel() / x.shape[0]
            x = x.view(-1, int(n_units))
            if self.print_toggle:
                print(f"Vectorized: {list(x.shape)}")

            # Linear layers
            x = F.leaky_relu(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.leaky_relu(self.fc2(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc3(x)
            if self.print_toggle:
                print(f"Final Output: {list(x.shape)}")

            return x

    model = EMNISTNet(print_toggle)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    return model, loss_fun, optimizer


model, loss_fun, optimizer = make_the_model(True)

X, y = next(iter(train_dl))
print(f"Y is: {y}")
y_hat = model(X)
loss = loss_fun(y_hat, torch.squeeze(y))

print(f"Output: {y_hat.shape} | Loss: {loss.item()}")

# training the model


def train_the_model():
    epochs = 30
    model, loss_fun, optimizer = make_the_model(False)

    model = model.to(device)

    train_loss = torch.zeros(epochs)
    test_loss = torch.zeros(epochs)
    train_acc = torch.zeros(epochs)
    test_acc = torch.zeros(epochs)

    for epoch_i in tqdm(range(epochs)):
        batch_loss = []
        batch_acc = []

        model.train()
        for X, y in tqdm(train_dl):
            X, y = X.to(device), y.to(device)

            y_hat = model(X)
            loss = loss_fun(y_hat, torch.squeeze(y))
            batch_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = torch.mean((torch.argmax(y_hat, axis=1) == y).float()).item()
            batch_acc.append(acc)

        train_acc[epoch_i] = np.mean(batch_acc)
        train_loss[epoch_i] = np.mean(batch_loss)

        model.eval()
        X, y = next(iter(test_dl))
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y_hat = model(X)
            loss = loss_fun(y_hat, torch.squeeze(y))
            acc = torch.mean((torch.argmax(y_hat, axis=1) == y).float()).item()

            test_acc[epoch_i] = acc
            test_loss[epoch_i] = loss.item()

    return train_loss, test_loss, train_acc, test_acc, model


train_loss, test_loss, train_acc, test_acc, model = train_the_model()

fig, ax = plt.subplots(1, 2, figsize=(12, 3))

ax[0].plot(train_loss, "s-", label="Train")
ax[0].plot(test_loss, "o-", label="Test")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss (MSE)")
ax[0].set_title("Model Loss")

ax[1].plot(train_acc, "s-", label="Train")
ax[1].plot(test_acc, "o-", label="Test")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy(%)")
ax[1].set_title(f"Final Test Accuracy: {test_acc[-1]: 0.2f}")
ax[1].legend()
plt.show()


X, y = next(iter(test_dl))

X = X.to(device)
y = y.to(device)

y_hat = model(X)

# rand_idx = np.random.choice(len(y), size=21, replace=False)

# fg, axs = plt.subplots(3, 7, figsize=(12, 6))

# for i, ax in enumerate(axs.flatten()):
#     idx = rand_idx[i]
#     image = np.squeeze(X[idx, 0, :, :]).cpu()
#     true_letter = character_categories[y[idx]]
#     pred_letter = character_categories[torch.argmax(y_hat, axis=1)[idx]]

#     cmap = "gray" if true_letter == pred_letter else "hot"

#     ax.imshow(image, cmap=cmap)
#     ax.set_title(f"True: {true_letter} | Pred: {pred_letter}", fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()

# print(y.cpu())
# print(y_hat.cpu())
C = skm.confusion_matrix(y.cpu(), torch.argmax(
    y_hat.cpu(), axis=1), normalize='true')

fig = plt.figure(figsize=(6, 6))
plt.imshow(C, "Blues", vmax=0.5)

plt.xticks(range(46), labels=character_categories) #change to 46
plt.yticks(range(46), labels=character_categories)
plt.title("Test Confusion Matrix")
plt.show()

# save the model
# definer the model in outer scope
class EMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.print_toggle = print_toggle

        # Conv1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(num_features=64)

        # Conv2
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3)
        # Input: number of channels
        self.bnorm2 = nn.BatchNorm2d(num_features=128)

        # Conv3
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3)
        # Input: number of channels
        self.bnorm3 = nn.BatchNorm2d(num_features=256)

        self.fc1 = nn.Linear(in_features=2*2*256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=47) #change here to 47

    def forward(self, x):
        # if self.print_toggle:
        #     print(f"Input: {list(x.shape)}")

        # First Block: conv -> max_pool -> bnorm -> relu
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.leaky_relu((self.bnorm1(x)))
        x = F.dropout(x, p=0.25, training=self.training)
        # if self.print_toggle:
        #     print(f"First Block: {list(x.shape)}")

        # Second Block: conv -> max_pool -> bnorm -> relu
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.leaky_relu((self.bnorm2(x)))
        x = F.dropout(x, p=0.25, training=self.training)
        # if self.print_toggle:
        #     print(f"Second Block: {list(x.shape)}")

        # Third Block: conv -> max_pool -> bnorm -> relu
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.leaky_relu((self.bnorm3(x)))
        x = F.dropout(x, p=0.25, training=self.training)
        # if self.print_toggle:
        #     print(f"Second Block: {list(x.shape)}")

        # Reshape for linear layer
        n_units = x.shape.numel() / x.shape[0]
        x = x.view(-1, int(n_units))
        # if self.print_toggle:
        #     print(f"Vectorized: {list(x.shape)}")

        # Linear layers
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        # if self.print_toggle:
        #     print(f"Final Output: {list(x.shape)}")

        return x

this_model = EMNISTModel()
torch.save(this_model, "./EMNISTBalanced.pt")
