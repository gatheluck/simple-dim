__all__ == [
    'LocalDiscriminator',
]

import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalDiscriminator(nn.Module):
    '''
    Implementation of concat-and-convolve local disctiminator.
    For detail, please see Table 7 in original paper. 
    '''
    def __init__(self, input_dim):
        super(LocalDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(self.input_dim, 512, kernel_size=1)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=1)
        self.conv3 = nn.Conv1d(512, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

