'''
Copyright 2020 - Kush Khosla, Robbie Jones, Rohan Sampath

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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


class CifarClassifier(nn.Module):
    def __init__(self, use_cuda=False):
        super(CifarClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.max = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

        self.device = torch.device('cuda:0' if self.use_cuda and torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max(x)
        x = F.relu(self.conv2(x))
        x = self.max(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        features_for_evaluator = self.fc2(x)
        x = F.relu(features_for_evaluator)
        x = self.fc3(x)
        return x, features_for_evaluator

    def train(self,
                train_dataset,
                eval_dataset,
                batch_size=16,
                number_epochs=8,
                lr=1e-3,
                momentum=.9,
                log_every=2000):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                                    shuffle=False)

        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
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
    cifarClassifier = CifarClassifier()

    summary(cifarClassifier, input_size=(3, 32, 32))

    x = torch.rand(3, 32, 32)
    y, _ = cifarClassifier.forward(torch.unsqueeze(x, dim=0))
    print('The input size is: ', x.size())
    print('The output size is: ', torch.squeeze(y).size())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_shape', default=False)
    parser.add_argument('--cuda', default=False)
    args = parser.parse_args()

    if args.test_shape:
        test_shapes()
    else:
        classifier = CifarClassifier(use_cuda=args.cuda)

        transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
        )


        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

        batch_size = 64
        log_every = 250
        number_epochs = 20
        training_results = classifier.train(trainset, testset, batch_size=batch_size,
                                            log_every=log_every, number_epochs=number_epochs)

        model_name = f'cifar_classifier_mlp_bs{batch_size}_log_every{log_every}_ne{number_epochs}'

        plt.plot(training_results[0], training_results[2])
        plt.savefig(model_name + '.png')

        torch.save(classifier.state_dict(), model_name + '.pt')

        plt.show()
