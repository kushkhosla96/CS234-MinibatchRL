'''
The point of this file is to see if the data evaluator actually learns anything
useful. To test that, we use the data evaluator to select the 'best' and 'worst'
training examples from our training dataset. We then train a classifier excluding
the best, and excluding the worst. The hope is as we exclude the top x% of examples,
training becomes much worse. If we exclude the bottom x% of examples, training should
be much better.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torchsummary import summary

import argparse

import numpy as np
import matplotlib.pyplot as plt

from CifarClassifier import CifarClassifier
from CifarDataEvaluator import CifarDataEvaluator
from CifarDataEvaluatorMLP import CifarDataEvaluatorMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluator_location', default=None)
    parser.add_argument('--classifier_location', default=None)
    parser.add_argument('--cuda', default=False)
    args = parser.parse_args()

    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')

    evaluator = CifarDataEvaluatorMLP()
    if args.evaluator_location is not None:
        evaluator.load_state_dict(torch.load(args.evaluator_location,
                                                map_location=device))
    evaluator = evaluator.to(device)

    classifier = CifarClassifier()
    if args.classifier_location is not None:
        classifier.load_state_dict(torch.load(args.classifier_location,
                                                map_location=device))
    classifier = classifier.to(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    print(type(trainset.data), trainset.data.shape, type(trainset.targets))

    '''
    evaluations_of_trainset = []
    with torch.no_grad():
        for data in trainset:
            inputs, labels = data[0], data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            _, features = classifier.forward(inputs)
            features = features.to(device)

            evaluator_values = evaluator.forward(features)
            print(evaluator_values)
    '''

    percentages = [.1, .2, .3, .4, .5]
    remove_low_value_accuracies = []
    remove_high_value_accuracies = []

if __name__ == '__main__':
    main()
