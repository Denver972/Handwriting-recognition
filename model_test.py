
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
        self.conv1 = nn.Conv2d(input_features, 28, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv15 = nn.Conv2d(28, 28, 3, 1)  # testing out this #28 to 30
        self.pool15 = nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1)  # testing this out
        self.conv2 = nn.Conv2d(28, 56, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # self.conv25 = nn.Conv2d(60, 60, 1, 4)  # test7
        # self.pool25 = nn.MaxPool2d(kernel_size=1, stride=1)  # test7
        self.conv3 = nn.Conv2d(56, 112, kernel_size=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=1)
        # self.conv4 = nn.Conv2d(120, 240, 1, 1)
        # self.pool4 = nn.MaxPool2d(1, 1)
        self.fc_1 = nn.Linear(112, 18)
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
    predicted = [p.numpy() for p in predicted]
    predicted = np.array(predicted, dtype=np.uint).flatten()

    return predicted


conv_model = torch.load("./Model9-AugmentedTraining.pt")
custom_test_data = MWINPDataset("1959CharactersTest3.csv", "./")
test_loader = DataLoader(custom_test_data, batch_size=1000000, shuffle=True)

predict = predicted_values(conv_model, test_loader)
# print(predict)

df = pd.read_csv("1959CharactersTest3.csv")


new_df = df.assign(PredClass=predict)
new_df.to_csv("CharactersTest3Model9Augmented.csv", index=False)

# acc = accuracy(conv_model, test_loader)
# print("Accuracy CNN: ", acc[0])
# print(f"Confusion matrix:\n{acc[1]}")
