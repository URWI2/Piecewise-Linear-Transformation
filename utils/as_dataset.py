from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle
import cloudpickle

from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

__author__ = 'Weiqi Ji'
__copyright__ = 'Copyright 2019, Weiqi Ji'
__credits__ = ['Weiqi Ji']
__license__ = ''
__version__ = '0.1.0'
__maintainer__ = 'Weiqi Ji'
__email__ = 'jiweiqi10@gmail.com'
__status__ = 'Dev'


class Dataset:
    def __init__(self, dataset_name=None, missing=None):
        self.dataset_name = dataset_name
        self.missing = missing

        self.num_classes = 0
        self.img_rows = 0
        self.img_cols = 0
        self.img_channels = 0
        self.input_shape = []

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.x_test_certain = []
        self.y_test = []
        self.pdf_data = []

        self.x_train_mean = 0

        if dataset_name:
            self.load_data()

    def decode_predictions(self, y):
        if self.dataset_name == 'wifi':
            class_list = ['0', '1', '2', '3']
            index = y.argmax()
            print('label: {}, score: {:.2f}'.format(class_list[index],
                                                    y.max()))
        if self.dataset_name == 'mnist':
            class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            index = y.argmax()
            print('label: {}, score: {:.2f}'.format(class_list[index],
                                                    y.max()))

        if self.dataset_name == 'cifar10':
            class_list = [
                'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                'ship', 'truck'
            ]
            index = y.argmax()
            print('label: {}, score: {:.2f}'.format(class_list[index],
                                                    y.max()))

        if self.dataset_name == 'cifar100':
            class_list = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed',
                'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge',
                'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar',
                'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
                'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
                'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree',
                'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
                'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
                'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
                'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                'telephone', 'television', 'tiger', 'tractor', 'train',
                'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
                'wolf', 'woman', 'worm'
            ]
            index = y.argmax()
            print('label: {}, score: {:.2f}'.format(class_list[index],
                                                    y.max()))

    def revert_input(self, x):
        if self.dataset_name in ['cifar10', 'cifar100']:
            return np.int8((x + self.x_train_mean) * 255)

        else:
            return np.int8(x)

    def load_data(self):

        with open(f'./{self.dataset_name}/data/{self.dataset_name}_data_test.npy', 'rb') as f:
            self.test_certain = np.load(f)
            self.y_test = np.load(f)
        with open(f'./{self.dataset_name}/data/{self.dataset_name}_data_{self.missing}.pkl', 'rb') as f:
            self.x_test = np.array(pickle.load(f))
        try:
            with open(f'./{self.dataset_name}/data/{self.dataset_name}_saved_pdfs_{self.missing}.pkl', 'rb') as f:
                self.pdf_data = pickle.load(f)
        except:
            with open(f'./{self.dataset_name}/data/{self.dataset_name}_saved_pdfs_{self.missing}.cp.pkl', 'rb') as f:
                self.pdf_data = cloudpickle.load(f)

        print(self.x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        # self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)
    
        self.num_classes = self.y_test.shape[1]
        self.img_rows = self.x_test.shape[1]
