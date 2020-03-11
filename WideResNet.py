'''
This uses the model from the following repo:
https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import argparse

import numpy as np
import matplotlib.pyplot as plt

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, num_classes=10, widen_factor=1, dropRate=0.0, use_cuda=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

        self.device = torch.device('cuda:0' if self.use_cuda and torch.cuda.is_available() else 'cpu')


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        features = self.relu(self.bn1(out))
        out = F.avg_pool2d(features, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), features


    def train(self,
                train_dataset,
                eval_dataset,
                batch_size=16,
                number_epochs=8,
                lr=1e-1,
                momentum=.9,
                nesterov=True,
                log_every=2000):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                                    shuffle=False)

        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
        cross_entropy = nn.CrossEntropyLoss()

        # ever log_every number of batches, we will output the number
        # of examples that have been used for training, up to this point
        number_examples_used = 0
        examples_used = []

        # every log_every number of batches, we will output the loss
        # over that batch.
        training_losses = []

        # every log_every number of batches, we will output the loss over
        # the eval set
        eval_accuracy = []

        for epoch in range(number_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):

                batch_inputs, batch_labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad()

                predictions, _ = self.forward(batch_inputs)
                predictions = predictions.to(self.device)
                loss = cross_entropy(predictions, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                number_examples_used += batch_inputs.size(0)


                if i % log_every == log_every - 1:
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for data in self.eval_loader:
                            eval_inputs, eval_labels = data[0].to(self.device), data[1].to(self.device)
                            predictions, _ = self.forward(eval_inputs)
                            predictions = predictions.to(self.device)
                            _, predicted = torch.max(predictions.data, 1)
                            total += eval_labels.size(0)
                            correct += (predicted == eval_labels).sum().item()

                    print('[%d, %5d, %d] loss: %.3f. test accuracy: %d %%' %
                          (epoch + 1, i + 1, number_examples_used,
                            running_loss / log_every, 100 * correct / total))


                    examples_used.append(number_examples_used)
                    training_losses.append(running_loss)
                    eval_accuracy.append(correct / total)

                    running_loss = 0.0

        with torch.no_grad():
            correct = 0
            total = 0
            for data in self.eval_loader:
                eval_inputs, eval_labels = data[0].to(self.device), data[1].to(self.device)
                predictions, _ = self.forward(eval_inputs)
                predictions = predictions.to(self.device)
                _, predicted = torch.max(predictions.data, 1)
                total += eval_labels.size(0)
                correct += (predicted == eval_labels).sum().item()

            examples_used.append(number_examples_used)
            training_losses.append(running_loss)
            eval_accuracy.append(correct / total)

        return examples_used, training_losses, eval_accuracy

def test_shapes():
    cifarClassifier = WideResNet()

    summary(cifarClassifier, input_size=(3, 32, 32))

    x = torch.rand(3, 32, 32)
    y, _ = cifarClassifier.forward(torch.unsqueeze(x, dim=0))
    print('The input size is: ', x.size())
    print('The output size is: ', torch.squeeze(y).size())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_shape', default=False)
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--trained_classifier_path', default=None)
    args = parser.parse_args()

    if args.test_shape:
        test_shapes()
    else:
        if args.trained_classifier_path is None:
            classifier = WideResNet(use_cuda=args.cuda)
        else:
            classifier = WideResNet(use_cuda=args.cuda)
            classifier.load_state_dict(torch.load(args.trained_classifier_path, map_location=classifier.device))

        transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
        )


        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

        batch_size = 256
        log_every = 50
        number_epochs = 1
        training_results = classifier.train(trainset, testset, batch_size=batch_size,
                                            log_every=log_every, number_epochs=number_epochs)

        model_name = f'wideresnet_classifier_mlp_bs{batch_size}_log_every{log_every}_ne{number_epochs}'

        plt.plot(training_results[0], training_results[2])
        plt.savefig(model_name + '.png')

        torch.save(classifier.state_dict(), model_name + '.pt')

        plt.show()
