# Imports
import torch.nn as nn
import torch.nn.functional as F

class DigitRecognitionCNN(nn.Module):
  def __init__(self):
    super(DigitRecognitionCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    self.dropout1 = nn.Dropout2d(0.25)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.dropout2 = nn.Dropout2d(0.25)
    self.flatten = nn.Flatten()
    self.fully_connected1 = nn.Linear(16 * 7 * 7, 128)
    self.dropout3 = nn.Dropout(0.5)
    self.fully_connected2 = nn.Linear(128, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)
    x = self.dropout2(x)
    x = self.flatten(x)
    x = F.relu(self.fully_connected1(x))
    x = self.dropout3(x)
    x = self.fully_connected2(x)
    return F.log_softmax(x, dim=1)