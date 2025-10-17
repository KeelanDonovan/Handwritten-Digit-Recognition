# Imports
import torch.nn as nn
import torch.nn.functional as F

class DigitRecogCNN(nn.Module):
  def __init__(self):
    super(DigitRecogCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.fully_connected1 = nn.Linear(32 * 3 * 3, 10)
    self.flatten = nn.Flatten()
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x, 2)
    x = self.flatten(x)
    x = self.fully_connected1(x)
    return F.log_softmax(x, dim=1)
    