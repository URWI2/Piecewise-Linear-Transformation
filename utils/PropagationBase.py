import os
import torch
from typing import List
import matplotlib.pyplot as plt
import pickle

class Model():
    '''
    NN model class that is used throughout the project
    '''
    def __init__(self, model_name, alpha=0.01):
        '''
        :param model_name: folder name in "saves"
        :param alpha: LeakyRelu-parameter, is zero in case of relu activation
        '''
        self.folder = os.path.join("saves", model_name)
        self.location = os.path.join(self.folder, "model.pt")
        assert os.path.exists(self.location), "No such Model"
        self.get_parameters()
        self.alpha = alpha

    def get_parameters(self):
        '''
        retrieves weights and biases from torch model saved as parameter dictionary
        '''
        self.w = []
        self.b = []
        param_dict = torch.load(self.location)
        self.num_layers = len({int(item.split("_")[1]) for item in param_dict})
        for i in range(self.num_layers):
            self.w.append(param_dict["weights_" + str(i)].tolist())
            self.b.append(param_dict["bias_" + str(i)].tolist())
        self.layer_sizes = [len(self.w[0][0])] + [len(item) for item in self.w]

    def activation(self, datum):
        '''
        activation function of the model
        '''
        for i in range(len(datum)):
            if datum[i] < 0:
                datum[i] = datum[i] * self.alpha

class Method():
    '''
    the ADF propagation inherits from this class
    '''
    def __init__(self):
        self.results = {"distribution": None, "other": None, }

    def initialize_object(self):
        raise NotImplementedError()

    def initialize_results(self, model):
        raise NotImplementedError()

    def load_results(self, obj):
        raise NotImplementedError()

    def transform_linear(self, w, b):
        raise NotImplementedError()

    def activation(self, func):
        raise NotImplementedError()

    def object(self):
        raise NotImplementedError()

    def infos(self):
        raise NotImplementedError()


    def snapshot(self, ax_list, color):
        raise NotImplementedError()

    def finalize(self):
        pass

    def save_results(self):
        raise NotImplementedError()

    def plot(self, ax):
        raise NotImplementedError()

    def plot_one(self, ax):
        raise NotImplementedError()
