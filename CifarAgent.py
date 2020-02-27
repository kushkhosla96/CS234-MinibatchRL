import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torchsummary import summary

from CifarClassifier import CifarClassifier
from CifarDataEvaluator import CifarDataEvaluator

class CifarAgent():
    def __init__(self, classifier=None, evaluator=None):
        self.classifier = classifier if classifier else CifarClassifier()
        self.evaluator = evaluator if evaluator else CifarDataEvaluator()

    def train(train_dataset,
                eval_dataset,
                large_batch_size=64,
                small_batch_size=16,
                eval_batch_size=16,
                number_epochs=8,
                inner_iteration=200,
                moving_average_window=20,
                classifier_lr=1e-3,
                evaluator_lr=1e-3):
        train_loader = DataLoader(train_dataset, batch_size=large_batch_size,
                                    shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size,
                                    shuffle=True)

        classifier_optimizer = optim.SGD(self.classifier.parameters(),
                                            lr=classifier_lr)
        evaluator_optimizer = optim.SGD(self.evaluator.parameters(),
                                            lr=evaluator_lr)

        cross_entropy = nn.CrossEntropyLoss()
        delta = 0
        total_eval_samples = eval_batch_size * len(eval_loader)

        for epoch in range(number_epochs):
            for i, data in enumerate(train_loader, 0):
                batch_inputs, batch_labels = data

                batch_hs = self.evaluator(batch_inputs)
                bern = Bernoulli(batch_hs)
                batch_s = bern.sample()

                for iteration in range(inner_iteration):
                    classifier_optimizer.zero_grad()

                    # this selects a random subset of the batch to use
                    indices_to_use = np.random.choice(large_batch_size,
                                                        small_batch_size)

                    mini_batch_inputs = batch_inputs[indices_to_use]
                    mini_batch_labels = batch_labels[indices_to_use]
                    mini_batch_s = batch_s[indices_to_use]

                    mini_batch_predictions = self.classifier(mini_batch_inputs)

                    mini_batch_losses = cross_entropy(mini_batch_predictions,
                                                        mini_batch_labels)
                    classifier_loss = torch.mean(mini_batch_s * mini_batch_losses)
                    classifier_loss.backward()
                    classifier_optimizer.step()

                # this gets the factor infront of grad(log(pi))
                with torch.no_grad():
                    classifier_validation_loss = 0
                    for j, eval_data in enumerate(eval_loader, 0):
                        eval_batch_inputs, eval_batch_labels = eval_data
                        eval_batch_predictions = self.classifier(eval_batch_inputs)
                        eval_batch_losses = cross_entropy(eval_batch_predictions,
                                                                eval_batch_labels)
                        to_add = torch.sum(eval_batch_losses) / total_eval_samples
                        classifier_validation_loss += to_add
                    grad_factor = classifier_validation_loss - delta

                evaluator_optimizer.zero_grad()
                log_pis = batch_s * torch.log(batch_hs) + \
                            (1 - batch_s) * torch.log(1 - batch_hs)
                evaluator_loss = grad_factor * torch.mean(log_pis)
                evaluator_loss.backward()
                evaluator_optimizer.step()

                delta = (moving_average_window - 1) / moving_average_window * delta + \
                        classifier_validation_loss / (moving_average_window)

def main():
    cifarAgent = CifarAgent()

    x = torch.rand(256, 3, 32, 32)
    sample_indices, classifications = cifarAgent.forward(x, 16)
    print('The input size is: ', x.size())
    print('The output size is: ', classifications.size())
    print('The number of sampled indices is: ', len(sample_indices))

if __name__ == '__main__':
    main()
