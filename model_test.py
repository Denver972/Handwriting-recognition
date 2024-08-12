
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

if torch.cuda.is_available():
    device = torch.device('cuda')  # Default CUDA device
else:
    device = torch.device('cpu')
#device = torch.device('cpu')
print(device)


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
        image /= 255.0
        # image.reshape(30, 30)
        image = np.expand_dims(image, axis=0)
        # print(image.shape)
        image = torch.tensor(image, dtype=torch.float32)
        # image =
        characters = self.MWINP_frame.iloc[idx, 2]
        # above extracts the class in number form
        # label = class_to_idx[characters]
        # label = torch.as_tensor(characters)
        sample = {'image': image,
                  'characters': torch.tensor(characters, dtype=torch.int32)}
        if self.transform:
            sample = self.transform(sample)

        return image, characters


# class ConvModel(nn.Module):
#     """
#     Basic CNN
#     """

#     def __init__(self,
#                  input_features):
#         """
#         Network levels,
#         """
#         super().__init__()
#         self.input_features = input_features
#         self.conv1 = nn.Conv2d(input_features, 28, kernel_size=5, stride=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
#         self.conv15 = nn.Conv2d(28, 28, 3, 1)  # testing out this #28 to 30
#         self.pool15 = nn.MaxPool2d(
#             kernel_size=3, stride=1, padding=1)  # testing this out
#         self.conv2 = nn.Conv2d(28, 56, kernel_size=2, stride=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
#         # self.conv25 = nn.Conv2d(60, 60, 1, 4)  # test7
#         # self.pool25 = nn.MaxPool2d(kernel_size=1, stride=1)  # test7
#         self.conv3 = nn.Conv2d(56, 112, kernel_size=1, stride=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=1, stride=1)
#         # self.conv4 = nn.Conv2d(120, 240, 1, 1)
#         # self.pool4 = nn.MaxPool2d(1, 1)
#         self.fc_1 = nn.Linear(112, 18)
#         #

#     def forward(self, x):
#         """
#         The forward method required by nn.Module base class.
#         connects the network
#         """

#         x = self.conv1(x)
#         x = torch.relu(x)
#         x = self.pool1(x)
#         x = self.conv15(x)  # test
#         x = torch.relu(x)  # test
#         x = self.pool15(x)  # test
#         x = self.conv2(x)
#         x = torch.relu(x)
#         x = self.pool2(x)
#         # x = self.conv25(x)  # test7
#         # x = torch.relu(x)  # test7
#         # x = self.pool25(x)  # test7
#         x = self.conv3(x)
#         x = torch.relu(x)
#         x = self.pool3(x)
#         # x = self.conv4(x)  # test 8
#         # x = torch.relu(x)  # test 8
#         # x = self.pool4(x)  # test 8
#         x = x.flatten(1, -1)
#         x = self.fc_1(x)
#         return F.log_softmax(x, dim=1)

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
        real, predict, labels=np.arange(10))
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


def predicted_values(model, data_loader):
    """
    OUTPUT: List of predicted characters
    """
    model.eval()
    predicted = []
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)

        out = model(x)
        pred = out.data.max(1, keepdim=False)[1]
        predicted.append(pred)
    predicted = [p.cpu().numpy() for p in predicted]
    predicted = np.array(predicted, dtype=np.uint).flatten()

    return predicted


test_model = torch.load("./EMNISTBalanced.pt")
test_model = test_model.to(device)
custom_test_data = MWINPDataset("test.csv", "./")
test_loader = DataLoader(custom_test_data, batch_size=1000000, shuffle=True)

predict = predicted_values(test_model, test_loader)
# print(predict)

df = pd.read_csv("test.csv")


new_df = df.assign(PredClass=predict)
new_df.to_csv("Win1EMNISTtest.csv", index=False)

# acc = accuracy(conv_model, test_loader)
# print("Accuracy CNN: ", acc[0])
# print(f"Confusion matrix:\n{acc[1]}")
