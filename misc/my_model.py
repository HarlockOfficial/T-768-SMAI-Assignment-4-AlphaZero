#
# The small ResNet we already trained and are going to use.
#
import torch
from torch import nn

input_size = 3 * 6 * 6
output_policy_size = 80


class MyResNet(nn.Module):

    class ResBlock(nn.Module):
        def __init__(self, channels, stride=1):
            super().__init__()
            self.conv_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn_1 = nn.BatchNorm2d(channels)
            self.relu_1 = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.conv_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn_2 = nn.BatchNorm2d(channels)
            self.relu_2 = nn.ReLU()

        def forward(self, x_in):
            residual = x_in
            x = self.conv_1(x_in)
            x = self.bn_1(x)
            x = self.relu_1(x)
            x = self.dropout(x)
            x = self.conv_2(x)
            x = self.bn_2(x)
            x += residual
            x = self.relu_2(x)
            return x

    class HeadValueBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv2d(channels, 3, kernel_size=1)
            self.bn = nn.BatchNorm2d(3)
            self.relu_1 = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc_1 = nn.Linear(input_size, 32)
            self.relu_2 = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.fc_2 = nn.Linear(32, 1)
            self.activation = nn.Tanh()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu_1(x)
            x = self.flatten(x)
            x = self.fc_1(x)
            x = self.relu_2(x)
            x = self.dropout(x)
            x = self.fc_2(x)
            x = self.activation(x)
            return x

    class HeadPolicyBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv2d(channels, 32, kernel_size=1)  # policy head
            self.bn = nn.BatchNorm2d(32)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(6 * 6 * 32, output_policy_size)
            # self.log_softmax = nn.LogSoftmax(dim=1)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.flatten(x)
            x = self.fc(x)
            # x = self.log_softmax(x).exp()   Do not use log-activation with the CrossEntropyLoss() function.
            return x


    def __init__(self):
        super().__init__()
        channels = 32
        self.conv = nn.Conv2d(3, channels, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.res_1 = MyResNet.ResBlock(channels)
        self.res_2 = MyResNet.ResBlock(channels)
        self.res_3 = MyResNet.ResBlock(channels)
        self.res_4 = MyResNet.ResBlock(channels)
        self.res_blocks = [self.res_1, self.res_2, self.res_3, self.res_4]
        self.value_head = MyResNet.HeadValueBlock(channels)
        self.policy_head = MyResNet.HeadPolicyBlock(channels)
        self.loss_value = nn.MSELoss()
        self.loss_policy = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        for block in self.res_blocks:
            x = block(x)
        x_v = self.value_head(x)
        x_p = self.policy_head(x)
        return x_v, x_p

    def loss_func(self, pred, label):
        (pred_v, pred_p) = pred
        (label_v, label_p) = label
        loss_v = self.loss_value(pred_v, label_v)
        loss_p = self.loss_policy(pred_p, label_p)
        return loss_v + loss_p
