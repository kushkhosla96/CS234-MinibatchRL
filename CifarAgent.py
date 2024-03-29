import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.distributions import Bernoulli
from torch.utils.data.sampler import WeightedRandomSampler

from torch.utils.data import DataLoader

from torchsummary import summary

import argparse

import numpy as np
import matplotlib.pyplot as plt

from CifarClassifier import CifarClassifier
from WideResNet import WideResNet
from CifarDataEvaluatorMLP import CifarDataEvaluatorMLP

class CifarAgent():
    def __init__(self, classifier=None, evaluator=None, cuda=False, name=""):
        self.classifier = classifier if classifier else WideResNet()
        self.evaluator = evaluator if evaluator else CifarDataEvaluatorMLP()

        self.cuda = cuda
        if self.cuda:
            self.classifier.cuda()
            self.evaluator.cuda()
        self.device = torch.device('cuda:0' if self.cuda and torch.cuda.is_available() else 'cpu')

        self.name = name

    def train(self,
                train_dataset,
                eval_dataset,
                large_batch_size=64,
                small_batch_size=16,
                eval_batch_size=16,
                number_epochs=8,
                inner_iteration=200,
                moving_average_window=20,
                classifier_lr=1e-2,
                classifier_momentum=.9,
                evaluator_lr=5e-2,
                evaluator_momentum=.9):

        self.train_loader = DataLoader(train_dataset, batch_size=large_batch_size,
                                    shuffle=True, pin_memory=self.cuda)
        self.eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size,
                                    shuffle=True, pin_memory=self.cuda)

        classifier_optimizer = optim.SGD(self.classifier.parameters(), lr=classifier_lr, momentum=classifier_momentum)
        evaluator_optimizer = optim.SGD(self.evaluator.parameters(), lr=evaluator_lr, momentum=evaluator_momentum)

        cross_entropy = nn.CrossEntropyLoss()
        delta = 0

        # every log_every number of large batches, we will output the loss
        # over the eval set
        number_examples_used = 0
        examples_used = []
        eval_accuracy = []

        batch_hs_means = []
        batch_hs_stds = []

        for epoch in range(number_epochs):
            for i, data in enumerate(self.train_loader, 0):
                batch_inputs, batch_labels = data[0].to(self.device), data[1].to(self.device)

                with torch.no_grad():
                    _, batch_features = self.classifier(batch_inputs)
                    batch_features = batch_features.to(self.device)
                batch_hs = self.evaluator(batch_features).to(self.device)

                batch_hs_means.append(batch_hs.mean())
                batch_hs_stds.append(batch_hs.std())

                bern = Bernoulli(batch_hs)
                batch_s = bern.sample().to(self.device)

                for iteration in range(inner_iteration):
                    classifier_optimizer.zero_grad()

                    maximum_possible_index = min(small_batch_size, batch_labels.size(0))

                    # this selects a random subset of the batch to use
                    indices_to_use = np.random.choice(maximum_possible_index,
                                                        small_batch_size)

                    mini_batch_inputs = batch_inputs[indices_to_use].to(self.device)
                    mini_batch_labels = batch_labels[indices_to_use].to(self.device)
                    mini_batch_hs = batch_hs[indices_to_use].to(self.device)

                    mini_batch_s = batch_s[indices_to_use].to(self.device)

                    mini_batch_predictions, mini_batch_features = self.classifier(mini_batch_inputs)
                    mini_batch_predictions = mini_batch_predictions.to(self.device)
                    mini_batch_features = mini_batch_features.to(self.device)

                    mini_batch_losses = cross_entropy(mini_batch_predictions,
                                                        mini_batch_labels)
                    mini_batch_losses = mini_batch_losses * mini_batch_hs.detach()

                    classifier_loss = torch.mean(mini_batch_s * mini_batch_losses).to(self.device)
                    classifier_loss.backward()
                    classifier_optimizer.step()

                    number_examples_used += mini_batch_inputs.size(0)

                # this gets the factor infront of grad(log(pi))
                with torch.no_grad():
                    number_eval_samples = 0
                    classifier_validation_loss = 0
                    correct = 0
                    for j, eval_data in enumerate(self.eval_loader, 0):
                        eval_batch_inputs, eval_batch_labels = eval_data[0].to(self.device), eval_data[1].to(self.device)
                        eval_batch_predictions, _ = self.classifier(eval_batch_inputs)
                        eval_batch_predictions = eval_batch_predictions.to(self.device)
                        eval_batch_losses = cross_entropy(eval_batch_predictions,
                                                                eval_batch_labels)
                        classifier_validation_loss += torch.sum(eval_batch_losses)

                        _, predicted = torch.max(eval_batch_predictions.data, 1)
                        number_eval_samples += eval_batch_labels.size(0)
                        correct += (predicted == eval_batch_labels).sum().item()

                    classifier_validation_loss /= number_eval_samples
                    grad_factor = classifier_validation_loss - delta
                    # print(grad_factor)

                    examples_used.append(number_examples_used)
                    accuracy = correct / number_eval_samples
                    eval_accuracy.append(accuracy)

                evaluator_optimizer.zero_grad()
                log_pis = batch_s * torch.log(batch_hs) + \
                            (1 - batch_s) * torch.log(1 - batch_hs)
                evaluator_loss = grad_factor * torch.mean(log_pis)
                evaluator_loss.backward()
                evaluator_optimizer.step()

                delta = (moving_average_window - 1) / moving_average_window * delta + \
                        classifier_validation_loss / (moving_average_window)

                print("[%d] test accuracy: %d %%" %
                        (number_examples_used, 100 * accuracy))

        plt.title("Average Data Evaluator Weights")
        plt.xlabel("Iteration")
        plt.ylabel("Weight")
        plt.ylim(0, 1)
        plt.plot(batch_hs_means)
        plt.savefig(self.name + "_average_weights.png")
        plt.close()

        plt.title("Data Evaluator Weights Standard Deviations")
        plt.xlabel("Iteration")
        plt.ylabel("Standard Deviation")
        plt.plot(batch_hs_stds)
        plt.savefig(self.name + "_weights_stds.png")
        plt.close()

        return examples_used, eval_accuracy

def test_shapes():
    cifarAgent = CifarAgent()

    x = torch.rand(256, 3, 32, 32)
    sample_indices, classifications = cifarAgent.forward(x, 16)
    print('The input size is: ', x.size())
    print('The output size is: ', classifications.size())
    print('The number of sampled indices is: ', len(sample_indices))

def test_cuda(cuda=False):
    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    # we call train so that cifarAgent creates the dataloaders
    # we pass it with number_epochs = 0 so that it doesn't actually do
    # any training, since we are just checking cuda status
    cifarAgent = CifarAgent(cuda=cuda)
    cifarAgent.train(train_dataset, test_dataset, number_epochs=1)

    images, labels = next(iter(cifarAgent.train_loader))
    print(f"Is cuda set? The answer is: {cuda}")
    print(f"Are the images on cuda? The answer is: {images.is_cuda}")
    print(f"Is the classifier on cuda? The answer is: {next(cifarAgent.classifier.parameters()).is_cuda}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_cuda', default=False)
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--trained_classifier_path', default=None)
    args = parser.parse_args()

    if args.test_cuda:
        test_cuda(args.cuda)
    else:
        if args.trained_classifier_path is not None:
            classifier = WideResNet(use_cuda=args.cuda)
            classifier.load_state_dict(torch.load(args.trained_classifier_path, map_location=classifier.device))
            print("LOADED THE MODEL")
        else:
            classifier = None

        transform = transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
        )

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

        lbs = 2048
        sbs = 128
        ebs = 64
        number_epochs = 4
        inner_iteration = 100
        moving_average_window = 15

        model_name = f'cifar_agent_wideresnet_mlp_lbs{lbs}_sbs{128}_ebs{64}_ne{number_epochs}_ii{inner_iteration}_maw{moving_average_window}'

        agent = CifarAgent(classifier=classifier, cuda=args.cuda, name=model_name)

        training_results = agent.train(trainset, testset, large_batch_size=lbs,
                                            small_batch_size=128, eval_batch_size=ebs, number_epochs=number_epochs,
                                            inner_iteration=inner_iteration, moving_average_window=moving_average_window)



        plt.plot(training_results[0], training_results[1])
        plt.savefig(model_name + '.png')

        torch.save(agent.classifier.state_dict(), model_name + '_classifier.pt')
        torch.save(agent.evaluator.state_dict(), model_name + '_evaluator.pt')

        plt.show()
