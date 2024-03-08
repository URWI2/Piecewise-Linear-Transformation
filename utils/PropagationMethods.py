from utils.utils import NodeCache, LayerCache
from utils.PropagationBase import Method
import numpy as np
import scipy.stats
import math
import itertools
from utils.utils import RDP
import random
import matplotlib.pyplot as plt

random.seed(123)

from utils.adf import LeakyReLU as ADFLeakyReLU
from utils.utils import new_ax

if __name__ == "__main__":
    import pylab


class ADF(Method):
    '''
    Assumed Density Filtering for neuron-wise propagation of gaussian distributions described by their means of and variances
    '''
    name = "ADF"

    def __init__(self, means, vars):
        '''
        :param means: list of means of input neurons
        :param vars: list of variances of input neurons
        '''
        super().__init__()
        self.cur_mean = means
        self.cur_var = vars
        self.act = ADFLeakyReLU().forward

    def initialize_object(self):
        pass

    def initialize_results(self, model):
        self.results["distribution"] = NodeCache(limit=2, model=model)

    def transform_linear(self, w, b):
        mean = len(b) * [0]
        var = len(b) * [0]
        for i in range(len(b)):
            for u in range(len(self.cur_mean)):
                mean[i] += w[i][u] * self.cur_mean[u]
                var[i] += w[i][u] ** 2 * self.cur_var[u]
            mean[i] += b[i]
        self.cur_mean = mean
        self.cur_var = var

    def activation(self, func=None):
        mean = []
        var = []
        for i in range(len(self.cur_mean)):
            #print(self.cur_mean[i], self.cur_var[i])
            m_relu, v_relu = self.act(self.cur_mean[i], self.cur_var[i])
            mean.append(m_relu)
            var.append(v_relu)
        self.cur_mean = mean
        self.cur_var = var

    def object(self):
        return self.cur_mean, self.cur_var

    def snapshot(self, ax_list, color=None):
        for u in range(len(self.cur_mean)):
            func = scipy.stats.norm(loc=self.cur_mean[u], scale=math.sqrt(self.cur_var[u]))
            distr = Distribution_1D(func.pdf, func.cdf, func.ppf(0.000001), func.ppf(1-0.000001))
            self.results["distribution"].save(distr, u)
            distr.plot(ax_list[u], color)
