from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from tqdm import tqdm_notebook as tqdm

from distribution import Distribution
from Polytopes import CPU, loadCPU
from Linear_Segments import Cube, getLinearSegments, get_restricted_cpu, loadModel, loadModel_AS

import numpy as np
import tensorflow as tf
import os
import math
from scipy.sparse.linalg import svds
from scipy.stats import truncnorm

from keras import backend as K

#import nnsubspace.visual.subspaceplot as subspaceplot

tf.compat.v1.disable_eager_execution()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

__author__ = 'Weiqi Ji'
__copyright__ = 'Copyright 2019, Weiqi Ji'
__credits__ = ['Weiqi Ji']
__license__ = ''
__version__ = '0.1.0'
__maintainer__ = 'Weiqi Ji'
__email__ = 'jiweiqi10@gmail.com'
__status__ = 'Dev'


class NNSubspace:
    def __init__(self,
                 model=None,
                 x=None,
                 inst_idx = None,
                 x_train_mean=None,
                 dataset_name=None,
                 dataset=None):

        self.verbose = 0

        self.model = model.model
        self.torch_model = model.torch_model
        self.x = x
        self.inst_idx = inst_idx
        self.x_train_mean = x_train_mean
        self.dataset=dataset

        self.dist = None      

        self.dataset_name = dataset_name

        self.tf_session = K.get_session()

        self.i_class = 0
        self.gradients = []

        self.num_eigenvalue = 10
        self.eigenvector = []
        self.eigenvalue = []

        self.num_gradient_mc = 0
        self.num_rs_mc = 0
        self.seed = None
        self.bool_clip = False
        self.sigma = 0

        self.x_samples_gradient = []
        self.x_samples_gradient_flat = []
        self.x_samples_rs = []

        self.y_samples_gradient = []
        self.y_samples_rs = []
        self.x_samples_rs_flat = []

        self.samples_gradient = []
        self.samples_gradient_flat = []

        self.xv = []
        self.poly1d = []
        self.poly1d_gradient = []

    def run(self, verbose=0):
        """
        Main function
        """

        self.verbose = verbose

        self.sampling()
        self.get_i_class()
        self.evaluate_gradient()
        self.evaluate_sample()
        self.svd()

    def sampling(self):
        """
        Generate samples
        """
        # print(self.dataset_name)
        _start_time = datetime.now()

        if self.x is None:
            raise Exception('x is not specified for sampling')

        if self.bool_clip:
            self.clip_gaussian()
        else:
            #self.gaussian()
            self.custom_distribution()

        self.x_samples_gradient = self.x_samples_rs[0:self.num_gradient_mc]

        self.x_samples_rs_flat = self.x_samples_rs.reshape(self.num_rs_mc, -1)
        self.x_samples_gradient_flat = self.x_samples_gradient.reshape(
            self.num_gradient_mc, -1)

        _time_elapsed = datetime.now() - _start_time

        if self.verbose > 0:
            print('Time elapsed (hh:mm:ss.ms) {} @sampling'.format(
                _time_elapsed))

    def sampling_setup(self,
                       num_gradient_mc=10,
                       num_rs_mc=10,
                       seed=None,
                       bool_clip=False,
                       sigma=0,
                       num_eigenvalue=10):
        """
        """

        self.num_gradient_mc = num_gradient_mc
        self.num_rs_mc = num_rs_mc
        self.seed = seed
        self.bool_clip = bool_clip
        self.sigma = sigma
        self.num_eigenvalue = num_eigenvalue

        if self.num_gradient_mc > self.num_rs_mc:
            raise Exception(
                'num_rs_mc = should ne larger than num_gradient_mc = {}'.
                format(self.num_rs_mc, self.num_gradient_mc))

    def get_i_class(self):

        if self.dataset_name == 'imagenet':
            self.i_class = self.model.predict(self.x * 255).argmax()
        else:
            self.i_class = self.model.predict(np.expand_dims(self.x, axis = 0)).argmax()

    def evaluate_gradient(self):
        # print("eval gradient")
        _start_time = datetime.now()

        self.gradients = K.gradients(self.model.output[0][self.i_class],
                                     self.model.input)
        self.samples_gradient = np.zeros_like(self.x_samples_gradient)

        for count, sample in enumerate(self.x_samples_gradient):
            if self.dataset_name == 'imagenet':
                evaluated_gradients = self.tf_session.run(self.gradients,
                                                          feed_dict={
                                                              self.model.input:
                                                              np.expand_dims(
                                                                  sample * 255,
                                                                  axis=0)
                                                          })
            else:
                evaluated_gradients = self.tf_session.run(self.gradients,
                                                          feed_dict={
                                                              self.model.input:
                                                              np.expand_dims(
                                                                  sample,
                                                                  axis=0)
                                                          })
            self.samples_gradient[count] = evaluated_gradients[0][0]

        _time_elapsed = datetime.now() - _start_time

        if self.verbose > 0:
            print('Time elapsed (hh:mm:ss.ms) {} @evaluate_gradient'.format(
                _time_elapsed))

        self.samples_gradient_flat = self.samples_gradient.reshape(
            self.num_gradient_mc, -1)

    def evaluate_sample(self):
        # print("eval sample")
        if self.dataset_name == 'imagenet':
            self.y_samples_rs = self.model.predict(self.x_samples_rs * 255,
                                                   batch_size=256,
                                                   verbose=0)[:, self.i_class]
        else:
            # self.y_samples_rs = self.model.predict(self.x_samples_rs,
            #                                        batch_size=64,
            #                                        verbose=0)[:, self.i_class]
            self.y_samples_rs = self.model.predict(self.x_samples_rs,
                                                  batch_size=64,
                                                  verbose=0)

    def svd(self):
        # print("start_svd")
        _, self.eigenvalue, self.eigenvector = svds(self.samples_gradient_flat,
                                                    k=self.num_eigenvalue,
                                                    which='LM',
                                                    tol=1e-16)

        self.eigenvalue = np.flip(self.eigenvalue, axis=0)
        self.eigenvector = np.flip(self.eigenvector.T, axis=1)

        self.xv = self.x_samples_rs_flat @ self.eigenvector
        for j in range(self.y_samples_rs.shape[1]):
            self.poly1d_gradient.append(np.poly1d(
                np.polyfit(self.xv[range(self.num_gradient_mc), 0],
                           self.y_samples_rs[range(self.num_gradient_mc), j], 2)))

            self.poly1d.append(np.poly1d(np.polyfit(self.xv[:, 0], self.y_samples_rs[:, j], 2)))

        # subspaceplot.eigenplot(self.eigenvalue, self.num_eigenvalue, figsize=None)

        # subspaceplot.summaryplot(self.xv[:, 0],
        #                          self.y_samples_rs,
        #                          self.poly1d,
        #                          figsize=None)

    def set_model(self, model):

        self.model = model

    def set_x(self, x, x_train_mean=None):

        self.x = x
        if x_train_mean is not None:
            self.x_train_mean = x_train_mean
            
    def custom_distribution(self):
        inst = np.squeeze(self.x)
        certainty_vector = list(inst)
        certainty_vector = [None if math.isnan(x) else x for x in certainty_vector]
        uncer_idx = [i for i,v in enumerate(certainty_vector) if v == None]
        cer_idx = [i for i,v in enumerate(certainty_vector) if v != None]
        
        t_model = loadModel_AS(self.torch_model)
        cube_dim = len(uncer_idx)
        cb = Cube(cube_dim) 

        if self.dataset_name in ['wifi']:
            # print("in custom distr normal")

            for d in range(cube_dim):
                cb.stretch(self.sigma*6, d+1)
                cb.translate(self.dataset.test_certain[0][uncer_idx[d]]-self.sigma*3, d+1)
                
            v = cb.points
            conn = cb.conn
            cpu_cb = CPU(v, conn)

            # res_cpu = get_restricted_cpu(model = t_model, cpu = cpu_cb, cert_vec = certainty_vector, output_vec = None)
            self.dist = Distribution(cpu = cpu_cb, model = t_model, distr_type = "normal", mean = self.dataset.test_certain[0][uncer_idx], cov = np.diag((self.sigma**2)*np.ones(cube_dim)))
        else:
            # print("in custom distr kde")
            for i, d in enumerate(uncer_idx):
                cb.stretch(1.2*(self.dataset.pdf_data[self.inst_idx][0].dataset[i].max() - self.dataset.pdf_data[self.inst_idx][0].dataset[i].min()), i+1)
                cb.translate(1.1*self.dataset.pdf_data[self.inst_idx][0].dataset[i].min() - 0.1*self.dataset.pdf_data[self.inst_idx][0].dataset[i].max(), i+1)
               
            v = cb.points
            conn = cb.conn
            cpu_cb = CPU(v, conn)
            # print("Start CPU")
            # res_cpu = get_restricted_cpu(model = t_model, cpu = cpu_cb, cert_vec = certainty_vector, output_vec = None)
            # print("CPU done")
            self.dist = Distribution(cpu = cpu_cb, model = t_model, distr_type = "custom", pdf = self.dataset.pdf_data[self.inst_idx][0])



        input_samps = self.dist.sample(self.num_rs_mc)
        if len(input_samps.shape) == 1:
            input_samps = np.reshape(input_samps, (len(input_samps),1))
        
        cert_samps = np.reshape([certainty_vector[i] for i in cer_idx]*len(input_samps), (len(input_samps), len(cer_idx)))
        sample_vec = np.zeros((len(input_samps), len(certainty_vector)))
        for idx, val in enumerate(certainty_vector):
            if idx in cer_idx:
                sample_vec[:,idx] = val
            else:
                sample_vec[:,idx] = input_samps[:, uncer_idx.index(idx)]
                
        self.x_samples_rs = sample_vec
       
    def gaussian(self):

        np.random.seed(self.seed)
        self.x_samples_rs = np.random.normal(
            loc=self.x,
            scale=self.sigma * np.ones_like(self.x),
            size=(self.num_rs_mc, self.x.shape[1], self.x.shape[2],
                  self.x.shape[3]))

    def clip_gaussian(self):

        np.random.seed(self.seed)
        self.x_samples_rs = np.zeros(
            (self.num_rs_mc,
             self.x.shape[1] * self.x.shape[2] * self.x.shape[3]))
        x_flat = self.x.flatten()
        x_train_mean_flat = self.x_train_mean.flatten()
        for i_sample in range(x_flat.shape[0]):
            self.x_samples_rs[:, i_sample] = truncnorm.rvs(
                (-x_train_mean_flat[i_sample] - x_flat[i_sample]) / self.sigma,
                (1 - x_train_mean_flat[i_sample] - x_flat[i_sample]) /
                self.sigma,
                loc=x_flat[i_sample],
                scale=self.sigma,
                random_state=None,
                size=self.num_rs_mc)
        self.x_samples_rs = self.x_samples_rs.reshape(self.num_rs_mc,
                                                      self.x.shape[1],
                                                      self.x.shape[2],
                                                      self.x.shape[3])

    def threecolumnplot(self, title=None):
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rc('legend', fontsize=14)
        matplotlib.rc('lines', linewidth=2)
        matplotlib.rc('axes', labelsize=16)
        matplotlib.rc('xtick', labelsize=12)
        matplotlib.rc('ytick', labelsize=12)

        fig, axs = plt.subplots(nrows=1,
                                ncols=3,
                                constrained_layout=True,
                                figsize=(12, 3.5))

        # eigenplot
        ax = axs[0]
        ax.semilogy(range(self.num_eigenvalue),
                    self.eigenvalue[0:self.num_eigenvalue]**2, '-o')
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue')

        # summary plot
        ax = axs[1]
        if self.x.max() > 1.1:
            ax.plot(self.xv[:, 0], self.y_samples_rs, 'o')
            ax.plot(np.sort(self.xv[:, 0]),
                    self.poly1d(np.sort(self.xv[:, 0])), '-')
        else:
            ax.plot(self.xv[:, 0] * 255, self.y_samples_rs, 'o')
            ax.plot(
                np.sort(self.xv[:, 0]) * 255,
                self.poly1d(np.sort(self.xv[:, 0])), '-')
            ax.plot(
                np.sort(self.xv[:, 0]) * 255,
                self.poly1d_gradient(np.sort(self.xv[:, 0])), '--')
        ax.set_xlabel(r'$w_1^T\xi$')
        ax.set_ylabel('Output')
        ax.set_title(title, fontsize=14)

        # histogram
        ax = axs[2]
        bin_number = 50
        ax.hist(self.y_samples_rs,
                bins=bin_number,
                density=True,
                histtype='step',
                label='MC',
                orientation="horizontal")
        ax.hist(self.poly1d(self.xv[:, 0]),
                bins=bin_number,
                density=True,
                histtype='step',
                label='RS',
                orientation="horizontal")
        ax.hist(self.poly1d_gradient(self.xv[:, 0]),
                bins=bin_number,
                density=True,
                histtype='step',
                label='RS_gradient',
                orientation="horizontal")
        ax.set_xlabel('PDF')
        ax.set_ylabel('Output')
        ax.legend(loc="best", frameon=False)

        plt.show()