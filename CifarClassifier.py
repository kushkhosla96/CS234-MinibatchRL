import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CifarClassifier(nn.Module):
    def __init__(self):
        super(CifarClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.max1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.max2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self,
                train_dataset,
                eval_dataset,
                batch_size=16,
                number_epochs=8,
                lr=1e-3,
                loss_every=2000):
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size,
                                    shuffle=True)

        optimizer = optim.SGD(self.parameters(), lr=lr)
        cross_entropy = nn.CrossEntropyLoss()

        for epoch in range(number_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                batch_inputs, batch_labels = data

                optimizer.zero_grad()

                predictions = self.forward(batch_inputs)
                loss = cross_entropy(predictions, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % loss_every == loss_every - 1:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / loss_every))
                    running_loss = 0.0


def main():
    cifarClassifier = CifarClassifier()

    summary(cifarClassifier, input_size=(3, 32, 32))

    x = torch.rand(3, 32, 32)
    y = cifarClassifier.forward(torch.unsqueeze(x, dim=0))
    print('The input size is: ', x.size())
    print('The output size is: ', torch.squeeze(y).size())

if __name__ == '__main__':
    main()
