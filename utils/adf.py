import operator
from collections import OrderedDict
from itertools import islice
import sys

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as tf
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _ConvTransposeMixin
from torch.nn.modules.utils import _pair
from utils.contrib.math import normpdf, normcdf

import math
import scipy.stats

class ReLU(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(ReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance):
        try:
            features_stddev = torch.sqrt(features_variance)
            div = features_mean / features_stddev
            pdf = normpdf(div)
            cdf = normcdf(div)
            outputs_mean = features_mean * cdf + features_stddev * pdf
            outputs_variance = (features_mean ** 2 + features_variance) * cdf \
                               + features_mean * features_stddev * pdf - outputs_mean ** 2
            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
            return outputs_mean, outputs_variance
        except TypeError:
            features_stddev = math.sqrt(features_variance)
            div = features_mean / features_stddev
            pdf = scipy.stats.norm(loc=0, scale=1).pdf(div)
            cdf = scipy.stats.norm(loc=0, scale=1).cdf(div)

            outputs_mean = features_mean * cdf + features_stddev * pdf
            outputs_variance = (features_mean ** 2 + features_variance) * cdf \
                               + features_mean * features_stddev * pdf - outputs_mean ** 2
            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
            return outputs_mean, outputs_variance

class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, keep_variance_fn=None):
        super(LeakyReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self._negative_slope = negative_slope

    def forward(self, features_mean, features_variance):
        try:
            features_stddev = torch.sqrt(features_variance)
            div = features_mean / features_stddev
            pdf = normpdf(div)
            cdf = normcdf(div)
            negative_cdf = 1.0 - cdf
            mu_cdf = features_mean * cdf
            stddev_pdf = features_stddev * pdf
            squared_mean_variance = features_mean ** 2 + features_variance
            mean_stddev_pdf = features_mean * stddev_pdf
            mean_r = mu_cdf + stddev_pdf
            variance_r = squared_mean_variance * cdf + mean_stddev_pdf - mean_r ** 2
            mean_n = - features_mean * negative_cdf + stddev_pdf
            variance_n = squared_mean_variance * negative_cdf - mean_stddev_pdf - mean_n ** 2
            covxy = - mean_r * mean_n
            outputs_mean = mean_r - self._negative_slope * mean_n
            outputs_variance = variance_r \
                               + self._negative_slope * self._negative_slope * variance_n \
                               - 2.0 * self._negative_slope * covxy
            if self._keep_variance_fn is not None:
                outputs_variance = self._keep_variance_fn(outputs_variance)
            return outputs_mean, outputs_variance
        except TypeError:
            features_stddev = math.sqrt(features_variance)
            div = features_mean / features_stddev
            pdf = scipy.stats.norm(loc=0, scale=1).pdf(div)
            cdf = scipy.stats.norm(loc=0, scale=1).cdf(div)
            negative_cdf = 1.0 - cdf
            mu_cdf = features_mean * cdf
            stddev_pdf = features_stddev * pdf
            squared_mean_variance = features_mean ** 2 + features_variance
            mean_stddev_pdf = features_mean * stddev_pdf
            mean_r = mu_cdf + stddev_pdf
            variance_r = squared_mean_variance * cdf + mean_stddev_pdf - mean_r ** 2
            mean_n = - features_mean * negative_cdf + stddev_pdf
            variance_n = squared_mean_variance * negative_cdf - mean_stddev_pdf - mean_n ** 2
            covxy = - mean_r * mean_n
            outputs_mean = mean_r - 0.01 * mean_n
            outputs_variance = variance_r + 0.01 * 0.01 * variance_n - 2.0 * 0.01 * covxy
            return outputs_mean, outputs_variance

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, keep_variance_fn=None):
        super(Linear, self).__init__()
        self._keep_variance_fn = keep_variance_fn
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = tf.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = tf.linear(inputs_variance, self.weight**2, None)
        if self._keep_variance_fn is not None:
            outputs_variance = self._keep_variance_fn(outputs_variance)
        return outputs_mean, outputs_variance
