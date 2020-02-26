import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CifarDataEvaluator(nn.Module):
    def __init__(self):
        super(CifarDataEvaluator, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.max2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = x.view(-1, 8 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    cifarDataEvaluator = CifarDataEvaluator()

    summary(cifarDataEvaluator, input_size=(3, 32, 32))

    x = torch.rand(3, 32, 32)
    y = cifarDataEvaluator.forward(torch.unsqueeze(x, dim=0))
    print('The input size is: ', x.size())
    print('The output size is: ', torch.squeeze(y, dim=0).size())

if __name__ == '__main__':
    main()
