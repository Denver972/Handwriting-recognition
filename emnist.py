# Playground to develop different models
import os
import cv2
import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
# from torchvision.datasets import EMNIST
# from skimage import io, transform
from sklearn import metrics
import matplotlib.pyplot as plt
# from PIL import Image
import numpy as np
import pandas as pd

# Choose divice to run the program on. Default to CUDA because it is much
# faster than the cpu
if torch.cuda.is_available():
    device = torch.device('cuda')  # Default CUDA device
else:
    device = torch.device('cpu')


# Prepare and load the data
n_epochs = 800
# batch_size_train = 100
# batch_size_test = 100
learning_rate = 0.0005
momentum = 0.5
log_interval = 10
loss_NLL = torch.nn.NLLLoss()
loss_MSE = torch.nn.MSELoss()
loss_CTC = torch.nn.CTCLoss(10, 10)

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# root_dir = "./MNIST"
# # root_dir = "./EMNIST/EMNIST/raw/gzip"

# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST(
#         root=root_dir, train=True, download=True,
#         transform=torchvision.transforms.Compose([
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(
#                 (0.1307,), (0.3081,))
#         ])),
#     batch_size=batch_size_train, shuffle=True)

# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST(
#         root_dir, train=False, download=True,
#         transform=torchvision.transforms.Compose([
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(
#                 (0.1307,), (0.3081,))
#         ])),
#     batch_size=batch_size_test, shuffle=True)

###### Custom Dataset######
data = pd.read_csv("./Training6DatasetAugmented.csv")

# lbl = data.Label
# # print(lbl)
# idx_to_class = {ix: label for ix, label in enumerate(lbl)}
# class_to_idx = {value: key for key, value in idx_to_class.items()}
# print(idx_to_class)


class MWINPDataset(Dataset):
    """MWINP handwritten dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
            Arguments:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
        """
        self.MWINP_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return (len(self.MWINP_frame))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.MWINP_frame.iloc[idx, 0])
        image = np.array(cv2.imread(img_name, 0))
        # image = image.transpose()
        image = image*1.
        image -= image.min()
        image = image/255.0  # image.max()  # 102  # 51  # image.max() #10 # 25.5  # image.max()
        # print(image)
        # image.reshape(30, 30)
        image = np.expand_dims(image, axis=0)
        # print(image.shape)
        image = torch.tensor(image, dtype=torch.float32)
        # image =
        characters = self.MWINP_frame.iloc[idx, 2]
        # [idx,2] for individual character recognition
        # label = class_to_idx[characters]
        # label = torch.as_tensor(characters)
        sample = {'image': image,
                  'characters': torch.tensor(characters, dtype=torch.int32)}
        if self.transform:
            sample = self.transform(sample)

        return image, characters


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image, characters = sample['image'], sample['characters']

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'landmarks': torch.from_numpy(characters)}

custom_train_data = MWINPDataset("Training6DatasetAugmented.csv", "./")
custom_test_data = MWINPDataset("Testing5.csv", "./")
# check if it has worked
# for i, sample in enumerate(custom_train_data):
#     print(i, custom_train_data[0], custom_train_data[1])

#     if i == 3:
#         break


custom_train_loader = DataLoader(
    custom_train_data, batch_size=64, shuffle=True)
# 64 batch size seems best
custom_test_loader = DataLoader(
    custom_test_data, batch_size=1000, shuffle=True)

# optimizer class. Currently most basic with gradient descent.


class GradientDescent():
    """
    A gradient descent optimizer.
    """

    def __init__(self,
                 parameters,
                 learning_rate):
        """
        Create a gradient descent optimizer.

        Arguments:
            parameters: Iterable providing the parameters to optimize.
            learning_rate: The learning rate to use for optimization.
        """
        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def zero_grad(self):
        for p in self.parameters:
            if not p.grad is None:
                p.grad.zero_()

    def step(self):
        """
        Perform a gradient descent step on parameters associated to this optimizer.
        """
        for p in self.parameters:
            p.data.add_(p.grad, alpha=-self.learning_rate)


class ConvModel(nn.Module):
    """
    Basic CNN
    """

    def __init__(self,
                 input_features):
        """
        Network levels,
        """
        super().__init__()
        self.input_features = input_features
        self.conv1 = nn.Conv2d(input_features, 30, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv15 = nn.Conv2d(30, 30, 3, 1)  # testing out this
        self.pool15 = nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1)  # testing this out
        self.conv2 = nn.Conv2d(30, 60, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.conv25 = nn.Conv2d(60, 60, 1, 4)  # test7
        # self.pool25 = nn.MaxPool2d(kernel_size=1, stride=1)  # test7
        self.conv3 = nn.Conv2d(60, 120, kernel_size=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=1)
        # self.conv4 = nn.Conv2d(120, 240, 1, 1)
        # self.pool4 = nn.MaxPool2d(1, 1)
        self.fc_1 = nn.Linear(120, 18)
        #

    def forward(self, x):
        """
        The forward method required by nn.Module base class.
        connects the network
        """

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv15(x)  # test
        x = torch.relu(x)  # test
        x = self.pool15(x)  # test
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        # x = self.conv25(x)  # test7
        # x = torch.relu(x)  # test7
        # x = self.pool25(x)  # test7
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        # x = self.conv4(x)  # test 8
        # x = torch.relu(x)  # test 8
        # x = self.pool4(x)  # test 8
        x = x.flatten(1, -1)
        x = self.fc_1(x)
        return F.log_softmax(x, dim=1)


def train_epoch(training_loader,
                validation_loader,
                model,
                loss,
                optimizer,
                device):
    """
    Again, this should be a useful docstring, but that would
    give away the answer for the exercise.
    """

    model.train()
    model.to(device)

    training_loss = 0.0
    n = len(training_loader)
    # print(training_loader[0])
    for i, (x, y) in enumerate(training_loader):
        # Set gradients to zero.
        optimizer.zero_grad()

        # check x,y types

        # print(f"Batch {i} - x type: {type(x)}")
        # print(x.size())
        # print(f"Batch {i} - y type: {type(y)}")
        # print(y.size())
        # Move input to device
        x = x.to(device)
        y = y.to(device)

        # Predict output, compute loss, perform optimizer step.
        y_pred = model(x)

        l = loss(y_pred, y)

        l.backward()

        optimizer.step()

        training_loss += l.item()
        print("Batch ({} / {}): Loss {:.2f}".format(i, n, l.item()), end="\r")

    training_loss /= n

    model.eval()
    validation_loss = 0.0
    n = len(validation_loader)

    for i, (x, y) in enumerate(validation_loader):
        # Move input to device
        # print(f"Batch {i} - x type: {type(x)}")
        # print(x.size())
        # print(f"Batch {i} - y type: {type(y)}")
        # print(y.size())
        x = x.to(device)
        y = y.to(device)

        # Predict output, compute loss, perform optimizer step.

        y_pred = model(x)
        l = loss(y_pred, y)

        validation_loss += l.item()
    validation_loss /= n

    model.to(torch.device("cpu"))

    return (training_loss, validation_loss)


def accuracy(model, validation_loader):
    model.eval()
    real = []
    predict = []
    correct = 0
    for i, (x, y) in enumerate(validation_loader):

        x = x.to(device)
        y = y.to(device)

        out = model(x)
        pred = out.data.max(1, keepdim=False)[1]
        correct += pred.eq(y.data.view_as(pred)).sum()

        real.append(y)
        predict.append(pred)

        # outputs = torch.sigmoid(outputs)
        # predict = (outputs).float()
        # print(predict)
        # targ.append(y.numpy())
        # for i in range(len(predict.tensor.detach().numpy())):
        #     pred.append(int(predict.tensor.detach().numpy()[i][0]))
    accu = 100. * correct / len(validation_loader.dataset)
    # targets = np.concatenate(targ)
    print(real)
    print(predict)
    real = [r.numpy() for r in real]
    predict = [p.numpy() for p in predict]
    real = np.array(real, dtype=np.uint).flatten()
    predict = np.array(predict, dtype=np.uint).flatten()
    print(real)
    print(predict)
    conf_mat = metrics.confusion_matrix(
        real, predict, labels=np.arange(17))
    # 14 for dataset 3
    # 18 for dataset4
    # 2 for dataset5
    # fp = 0
    # tp = 0
    # tn = 0
    # fn = 0
    # for i in range(len(targets)):
    #     if targets[i] == pred[i]:
    #         tp += 1
    #     elif targets[i] != pred[i]:
    #         tn += 1

    # a = (tp + tn)/(fp + tp + tn + fn)
    # print(pred)
    return (accu, conf_mat)


train_loss = np.zeros(n_epochs)
valid_loss = np.zeros(n_epochs)
conv_model = ConvModel(input_features=1)
for epoch in range(1, n_epochs+1):
    # constant learning rate
    if epoch < n_epochs/2:
        lr = learning_rate
    else:
        lr = learning_rate/10

    output = train_epoch(training_loader=custom_train_loader,
                         validation_loader=custom_test_loader,
                         model=conv_model,
                         loss=loss_NLL,
                         optimizer=torch.optim.SGD(params=conv_model.parameters(), lr=learning_rate,
                                                   momentum=0.9),
                         #  optimizer=GradientDescent(
                         #      parameters=conv_model.parameters(), learning_rate=lr),
                         device=device)
    train_loss[epoch-1] = output[0]
    valid_loss[epoch-1] = output[1]
    loss_difference = valid_loss[epoch-1] - train_loss[epoch-1]

    print(
        f"epoch {epoch}, training_loss {output[0]}, validation_loss {output[1]}")
    if valid_loss[epoch-1] < 1.0 and loss_difference > 0.5:
        break
acc = accuracy(conv_model, custom_test_loader)
print("Accuracy CNN: ", acc[0])
print(f"Confusion matrix:\n{acc[1]}")

torch.save(conv_model, "./Model9-AugmentedTraining.pt")
plt.plot(range(1, n_epochs+1), train_loss)
plt.plot(range(1, n_epochs+1), valid_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss aginst Epoch Model 9: Augmented Training")
plt.legend(["Training", "Validation"])
plt.savefig("Model9AugmentedTraining.png")
plt.show()
