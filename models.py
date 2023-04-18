import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


    

class MLPRegressor(nn.Module):
    '''
    embedding 태워서 input으로 넣도록 수정 필요
    '''
    def __init__(self, input_size=24, hidden_size=64, output_size=2, drop_out=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        # self.fc3 = nn.Linear(hidden_size, output_size, bias=True)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_out(F.relu(self.fc2(x)))
        # x = F.relu(self.fc3(x))
        return x


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size=24, out_channels=2):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        return x


class SVR(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass