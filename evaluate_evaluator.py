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

from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchsummary import summary

import argparse

import numpy as np
import matplotlib.pyplot as plt

from CifarClassifier import CifarClassifier
from CifarDataEvaluator import CifarDataEvaluator
from CifarDataEvaluatorMLP import CifarDataEvaluatorMLP

class CifarSubset(Dataset):
    def __init__(self, images, targets, transform=None):
        super(CifarSubset, self).__init__()
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]


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

    evaluations_of_trainset = torch.tensor([], requires_grad=False)
    train_loader = DataLoader(trainset, batch_size=16, shuffle=False)

    with torch.no_grad():
        for i, data in enumerate(train_loader, 0):
            if i == 2:
                break

            inputs, labels = data[0], data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            _, features = classifier.forward(inputs)
            features = features.to(device)

            evaluator_values = torch.squeeze(evaluator.forward(features))
            evaluations_of_trainset = torch.cat((evaluations_of_trainset, evaluator_values))

    worst_to_best_indices = evaluations_of_trainset.argsort()

    percentages = [0, .1, .2, .3, .4, .5]
    remove_low_value_accuracies = []
    remove_high_value_accuracies = []

    for percentage in percentages:
        number_of_examples_to_remove = int(percentage * len(worst_to_best_indices))

        indices_to_keep = worst_to_best_indices[number_of_examples_to_remove:]
        images_after_removing_low_value = trainset.data[indices_to_keep]
        targets_after_removing_low_value = np.array(trainset.targets)[indices_to_keep]
        cifar_data_after_removing_low_value = CifarSubset(images_after_removing_low_value,
                                                            targets_after_removing_low_value,
                                                            transform=transform)

        classifier_without_low_value = CifarClassifier()
        training_results = classifier_without_low_value.train(cifar_data_after_removing_low_value,
                                                                testset, number_epochs=3)
        remove_low_value_accuracies.append(training_results[2][-1])

        indices_to_keep = worst_to_best_indices[:len(worst_to_best_indices) - number_of_examples_to_remove]
        images_after_removing_high_value = trainset.data[indices_to_keep]
        targets_after_removing_high_value = np.array(trainset.targets)[indices_to_keep]
        cifar_data_after_removing_high_value = CifarSubset(images_after_removing_high_value,
                                                            targets_after_removing_high_value,
                                                            transform=transform)

        classifier_without_high_value = CifarClassifier()
        training_results = classifier_without_high_value.train(cifar_data_after_removing_high_value,
                                                                testset,
                                                                number_epochs=3)
        remove_high_value_accuracies.append(training_results[2][-1])


    print(remove_low_value_accuracies, remove_high_value_accuracies)

if __name__ == '__main__':
    main()
