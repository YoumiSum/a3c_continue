import numpy as np
import pandas as pd

import torch
import torch.nn as nn


class STRG(torch.nn.Module):
    def __init__(self, num_inputs=4, hidden_size=128, action_shape=4):
        super(STRG, self).__init__()
        self.hidden_size = hidden_size

        # batchx4x224x224 -> batchx8x112x112
        self.conv1_1 = nn.Conv2d(num_inputs, 8, kernel_size=3, stride=2, padding=1)
        self.lrelu1_1 = nn.LeakyReLU(0.1)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.lrelu1_2 = nn.LeakyReLU(0.1)

        # batchx8x112x112 -> batchx16x56x56
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.lrelu2_1 = nn.LeakyReLU(0.1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.lrelu2_2 = nn.LeakyReLU(0.1)

        # batchx16x56x56 -> batchx32x28x28
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.lrelu3_1 = nn.LeakyReLU(0.1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.lrelu3_2 = nn.LeakyReLU(0.1)

        # batchx32x28x28 -> batchx64x15x15
        self.conv4 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=1)
        self.lrelu4 = nn.LeakyReLU(0.1)

        # batchx64x15x15 -> batchx64x8x8
        self.conv5 = nn.Conv2d(64, 64, kernel_size=1, stride=2)
        self.lrelu5 = nn.LeakyReLU(0.1)

        self.lstm = nn.LSTMCell(64*8*8, self.hidden_size)

        self.critic = nn.Linear(self.hidden_size, 1)
        self.actor_mean = nn.Sequential(
                            nn.Linear(self.hidden_size, 64*8),
                            nn.LeakyReLU(0.3),
                            nn.Linear(64*8, 64*4),
                            nn.LeakyReLU(0.3),
                            nn.Linear(64 * 4, 128),
                            nn.LeakyReLU(0.3),
                            nn.Linear(128, action_shape),
                            nn.Tanh(),
        )
        self.actor_var = nn.Sequential(
                            nn.Linear(self.hidden_size, 64*8),
                            nn.LeakyReLU(0.3),
                            nn.Linear(64*8, 64*4),
                            nn.LeakyReLU(0.3),
                            nn.Linear(64 * 4, 128),
                            nn.LeakyReLU(0.3),
                            nn.Linear(128, action_shape),
                            nn.Tanh(),
        )

    def forward(self, x, hx, cx):
        x = self.lrelu1_1(self.conv1_1(x))
        x = self.lrelu1_2(self.conv1_2(x))
        x = self.lrelu2_1(self.conv2_1(x))
        x = self.lrelu2_2(self.conv2_2(x))
        x = self.lrelu3_1(self.conv3_1(x))
        x = self.lrelu3_2(self.conv3_2(x))
        x = self.lrelu4(self.conv4(x))
        x = self.lrelu5(self.conv5(x))
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return (
            self.critic(x),
            self.actor_mean(x), self.actor_var(x),
            hx, cx,
        )

class CNN(torch.nn.Module):
    def __init__(self, num_inputs=4, hidden_size=128, action_shape=4):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size

        # batchx4x224x224 -> batchx8x112x112
        self.conv1_1 = nn.Conv2d(num_inputs, 8, kernel_size=3, stride=2, padding=1)
        self.lrelu1_1 = nn.LeakyReLU(0.1)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.lrelu1_2 = nn.LeakyReLU(0.1)

        # batchx8x112x112 -> batchx16x56x56
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.lrelu2_1 = nn.LeakyReLU(0.1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.lrelu2_2 = nn.LeakyReLU(0.1)

        # batchx16x56x56 -> batchx32x28x28
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.lrelu3_1 = nn.LeakyReLU(0.1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.lrelu3_2 = nn.LeakyReLU(0.1)

        # batchx32x28x28 -> batchx64x15x15
        self.conv4 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=1)
        self.lrelu4 = nn.LeakyReLU(0.1)

        # batchx64x15x15 -> batchx64x8x8
        self.conv5 = nn.Conv2d(64, 64, kernel_size=1, stride=2)
        self.lrelu5 = nn.LeakyReLU(0.1)

        self.lstm = nn.LSTMCell(64*8*8, self.hidden_size)

        self.critic = nn.Linear(self.hidden_size, 1)
        self.actor_mean = nn.Sequential(
                            nn.Linear(self.hidden_size, 64*8),
                            nn.LeakyReLU(0.3),
                            nn.Linear(64*8, 64*4),
                            nn.LeakyReLU(0.3),
                            nn.Linear(64 * 4, 128),
                            nn.LeakyReLU(0.3),
                            nn.Linear(128, action_shape),
                            nn.Tanh(),
        )
        self.actor_var = nn.Sequential(
                            nn.Linear(self.hidden_size, 64*8),
                            nn.LeakyReLU(0.3),
                            nn.Linear(64*8, 64*4),
                            nn.LeakyReLU(0.3),
                            nn.Linear(64 * 4, 128),
                            nn.LeakyReLU(0.3),
                            nn.Linear(128, action_shape),
                            nn.Tanh(),
        )
        self.acm = nn.Sequential(
            nn.Linear(self.hidden_size, 64 * 8),
            nn.LeakyReLU(0.3),
            nn.Linear(64 * 8, 64 * 4),
            nn.LeakyReLU(0.3),
            nn.Linear(64 * 4, 128),
            nn.LeakyReLU(0.3),
            nn.Linear(128, action_shape),
            nn.Tanh(),
        )
        self.acv = nn.Sequential(
            nn.Linear(self.hidden_size, 64 * 8),
            nn.LeakyReLU(0.3),
            nn.Linear(64 * 8, 64 * 4),
            nn.LeakyReLU(0.3),
            nn.Linear(64 * 4, 128),
            nn.LeakyReLU(0.3),
            nn.Linear(128, action_shape),
            nn.Tanh(),
        )

        self.acm1 = nn.Sequential(
            nn.Linear(self.hidden_size + action_shape * 2, 128),
            nn.LeakyReLU(0.3),
            nn.Linear(128, action_shape),
            nn.Tanh(),
        )
        self.acv1 = nn.Sequential(
            nn.Linear(self.hidden_size + action_shape * 2, 128),
            nn.LeakyReLU(0.3),
            nn.Linear(128, action_shape),
            nn.Tanh(),
        )

    def forward(self, x, hx, cx):
        x = self.lrelu1_1(self.conv1_1(x))
        x = self.lrelu1_2(self.conv1_2(x))
        x = self.lrelu2_1(self.conv2_1(x))
        x = self.lrelu2_2(self.conv2_2(x))
        x = self.lrelu3_1(self.conv3_1(x))
        x = self.lrelu3_2(self.conv3_2(x))
        x = self.lrelu4(self.conv4(x))
        x = self.lrelu5(self.conv5(x))
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        strg_m, strg_v = self.actor_mean(x), self.actor_var(x)
        m, v = self.acm(x), self.acv(x)
        std = torch.concat((m, strg_m, x), dim=-1)
        var = torch.concat((strg_v, v, x), dim=-1)

        return (
            self.critic(x),
            self.acm1(std), self.acv1(var),
            hx, cx,
        )


if __name__ == '__main__':
    import pickle

    with open('../../temp.pickle', 'rb') as f:
        loaded_data = pickle.load(f)

    loaded_data = torch.tensor(loaded_data)
    hidden_size = 128
    model = CNN(num_inputs=4, hidden_size=hidden_size)
    hx = torch.zeros((1, hidden_size))
    cx = torch.zeros((1, hidden_size))

    value, action, hx, cx = model(loaded_data, hx, cx)
