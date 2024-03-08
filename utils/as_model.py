from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras import backend as K

import torch


class NNModel:
    def __init__(self, dataset_name='mnist', model_id=0):
        self.model_file = []
        self.model = []
        self.torch_model = []

        self.model_file_dict = {}
        self.model_file_dict_mnist = {}
        self.model_file_dict_cifar10 = {}
        self.model_file_dict_cifar100 = {}
        self.model_file_dict_imagenet = {}

        self.tf_session = K.get_session()
        self.set_model(dataset_name, model_id)

    def set_model(self, dataset_name, model_id):
        self.model_file_dict_mnist = {
            '0': 'models/model_mnist_cnn_softplus.h5',
            '1': 'model1'
        }
        self.model_file_dict_cifar10 = {
            '0': 'models/cifar10_ResNet20v2_model.189.h5',
            '1': 'model1'
        }
        self.model_file_dict_cifar100 = {
            '0': 'models/cifar100_ResNet20v2_model.090.h5',
            '1': 'model1'
        }
        self.model_file_dict_imagenet = {
            '0': 'models/model_resnet50_imagenet.h5',
            '1': 'model1'
        }
        self.model_file_dict_else = {
            '0': f'./{dataset_name}/model_{dataset_name}.pt'
        }

        self.model_file_dict = {
            'mnist': self.model_file_dict_mnist,
            'cifar10': self.model_file_dict_cifar10,
            'cifar100': self.model_file_dict_cifar100,
            'imagenet': self.model_file_dict_imagenet,
            'else': self.model_file_dict_else
        }

        try:
            self.model_file = self.model_file_dict['else'][model_id]
        except:
            print('Error: dataset {} and model_id {} is not avaliable'.format(
                dataset_name, model_id))

        try:
            self.model = load_model(self.model_file)
        except:
            # print("torch model")
            weight_dict = torch.load(self.model_file)
         
            
            layer_list = []
            weight_list = []
            nr_layers = int(len(weight_dict.keys())/2)
            for l in range(nr_layers):
                weight_list.append(np.transpose(weight_dict['weights_{}'.format(l)]))
                weight_list.append(weight_dict['bias_{}'.format(l)])
                if l == 0:
                    layer_list.append(tf.keras.layers.Dense(weight_dict['weights_{}'.format(l)].shape[0], activation='relu', input_shape=(weight_dict['weights_{}'.format(l)].shape[1],)))
                elif l == nr_layers-1:
                    layer_list.append(tf.keras.layers.Dense(weight_dict['weights_{}'.format(l)].shape[0]))
                else:
                    layer_list.append(tf.keras.layers.Dense(weight_dict['weights_{}'.format(l)].shape[0], activation='relu'))
            keras_model = tf.keras.models.Sequential(layer_list)
            # print(keras_model.summary())
            
            keras_model.set_weights(weight_list)
            
            self.model = keras_model
            self.torch_model = weight_dict