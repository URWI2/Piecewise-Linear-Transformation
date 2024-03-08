from __future__ import absolute_import
from __future__ import print_function

import logging
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as tf

from contrib.distributions import Dirichlet
from contrib.distributions import SmoothOneHot



class ADFSoftmax(nn.Module):
    def __init__(self, label_smoothing= 0.001, random_off_targets=True):
        super(ADFSoftmax, self).__init__()
        self._log_bias_c1 = Parameter(-torch.ones(1, 1) * 0.1, requires_grad=True)
        self._log_bias_c2 = Parameter(-torch.ones(1, 1) * 0.1, requires_grad=True)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, outputs_mean, outputs_variance):

        c1 = tf.softplus(self._log_bias_c1)
        c2 = tf.softplus(self._log_bias_c2)
        #print("hier", c1, c2)
        # Vc1c2
        #print("means:", outputs_mean)
        #print("variances:", outputs_variance[0])
        mu = self._softmax(outputs_mean)
        #print("mu:", mu)
        stddev = torch.sqrt(torch.sum(mu * outputs_variance, dim=1, keepdim=True))
        #stddev = torch.sqrt(torch.sum(mu ** 2 * outputs_variance, dim=1, keepdim=True))
        s = 1.0 / (1e-4  + c1 + c2 * stddev)
        #print("s", s)
        #print("mu", mu)
        alpha =  mu * s
        #print("alpha:",alpha)
        #print("mu:", mu[0])
        #print("s", s)

        predictions = alpha / alpha.sum(dim=-1, keepdim=True)
        log_predictions = torch.log(predictions)
        return mu, alpha


class DirichletLoss(nn.Module):
    def __init__(self, label_smoothing= 0.001, random_off_targets=True):
        super(DirichletLoss, self).__init__()
        self._smoothed_onehot = SmoothOneHot(
            label_smoothing=label_smoothing, random_off_targets=random_off_targets)
        self._dirichlet = Dirichlet(argmax_smoothing=0.5)

    def forward(self, alpha, target):

        #print(alpha)
        # convert to smoothed onehot labels
        num_classes = 2
        smoothed_onehot = Variable(
            self._smoothed_onehot(target.data, num_classes=num_classes), requires_grad=False)
        #print(smoothed_onehot)
        total_loss = - self._dirichlet(alpha, smoothed_onehot).mean()
        return total_loss
