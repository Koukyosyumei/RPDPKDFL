import numpy as np
import torch
import torch.nn as nn


class LMAE(nn.Module):
    def __init__(self, channel=3, height=64, width=64):
        super(LMAE, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

        self.fc1 = nn.Linear(channel * height * width, 1000)
        self.fc2 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, 1000)
        self.fc4 = nn.Linear(1000, channel * height * width)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        x = x.reshape(batch_size, (self.channel, self.height, self.width))
        return x


class InvLM(nn.Module):
    def __init__(self, input_dim=10, output_shape=(1, 28, 28), channel=1):
        super(InvLM, self).__init__()
        self.output_shape = output_shape
        self.fc = nn.Linear(input_dim, np.prod(output_shape))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.fc(x).reshape((batch_size,) + self.output_shape)


class InvCNN(nn.Module):
    def __init__(self, input_dim=40, output_shape=None, channel=3):
        super(InvCNN, self).__init__()
        self.ct1 = nn.ConvTranspose2d(input_dim, 1024, (4, 4), stride=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(
            1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.ct2 = nn.ConvTranspose2d(1024, 512, (4, 4), stride=(2, 2), padding=(1, 1))
        self.bn2 = torch.nn.BatchNorm2d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.ct3 = nn.ConvTranspose2d(512, 256, (4, 4), stride=(2, 2), padding=(1, 1))
        self.bn3 = torch.nn.BatchNorm2d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.ct4 = nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2), padding=(1, 1))
        self.bn4 = torch.nn.BatchNorm2d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.ct5 = nn.ConvTranspose2d(
            128, channel, (4, 4), stride=(2, 2), padding=(1, 1)
        )

    def forward(self, x):
        x = self.ct1(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.ct2(x)
        x = self.bn2(x)
        x = torch.tanh(x)
        x = self.ct3(x)
        x = self.bn3(x)
        x = torch.tanh(x)
        x = self.ct4(x)
        x = self.bn4(x)
        x = torch.tanh(x)
        x = self.ct5(x)
        x = torch.tanh(x)
        return x


def get_invmodel_class(invmodel_type):
    if invmodel_type == "InvCNN":
        model_class = InvCNN
    elif invmodel_type == "InvLM":
        model_class = InvLM
    else:
        raise NotImplementedError(f"{invmodel_type} is not supported.")
    return model_class
