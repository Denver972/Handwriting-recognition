# Playground to develop different models
import torch
import torchvision
import torch.nn as nn


# Choose divice to run the program on. Default to CUDA because it is much
# faster than the cpu
if torch.cuda.is_available():
    device = torch.device('cuda')  # Default CUDA device
else:
    device = torch.device('cpu')


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
        self.conv1 = nn.Conv2d(input_features, 16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(2304, 1)

    def forward(self, x):
        """
        The forward method required by nn.Module base class.
        connects the 
        """

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = x.flatten(1, -1)
        x = self.fc_1(x)
        return x
