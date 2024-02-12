#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0


import collections
import tensorflow as tf
from tensorflow import math as tfm
from neural_networks import NN_utils
from neural_networks import networks


class GaussianModel(object):
    """
    Gaussian neural network
    """

    def __init__(self, n_input, n_output, n_layers, name=None, nonlinearity=tf.nn.leaky_relu, sig_lim=10, constrain=False):
        """
        Initialisation. This function should accept the external parameters, such as input and output sizes.

        :param n_input: dimensionality of input
        :param n_output: dimensionality of output
        :param n_layers: vector or list of dimensionality for each hidden layers, e.g. [100,50,50]
        :param name: name to assign to the Gaussian model
        :param nonlinearity: non-linearity to use (default is leaky relu)
        :param sig_lim: range to impose to the output log_sig_sq to avoid divergence
        :param constrain: whether to force the output mean to be between 0 and 1

        """

        # casting the inputs as class global variables.
        self.n_input = n_input
        self.n_output = n_output
        self.n_layers = n_layers
        self.name = name
        self.bias_start = 0.0
        self.sig_lim = sig_lim
        self.constrain = constrain
        self.nonlinearity = nonlinearity

        # now we initialise the weights and set them as a global variable
        network_weights = self._create_weights()
        self.weights = network_weights

    def compute_moments(self, x):
        """
        compute moments of output Gaussian distribution from inputs.

        :param x: batch of inputs [n_batch x dimensions]

        :return: mean and log variance of Gaussian distribution.
        """

        # pass x through a fully connected network:
        hidden_post = networks.fc_network(x, self.weights, tf.shape(self.n_layers)[0], self.nonlinearity, id=0)

        # We take the output of the last network and pass it through two linear matrices to get mu and log_sigma_square
        mu = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, id=1)
        log_sig_sq = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, id=2)

        # constrain the mean of the output distribution to be between 0 and 1 (set to false if not)
        if self.constrain:
            mu = tf.nn.sigmoid(mu)

        # constrain log_sigma_square to be between -sig_lim and +sig_lim
        log_sig_sq = self.sig_lim * (tf.nn.sigmoid(log_sig_sq / self.sig_lim) - 0.5)

        return mu, log_sig_sq

    def _create_weights(self):
        """
        Initialise weights.
        """

        # first, we initialise an empty ordered dictionary
        all_weights = collections.OrderedDict()

        # we make the weights for the fully connected network taking x:
        all_weights = networks.fc_make_weights(all_weights, self.n_input, self.n_layers, id=0)

        # We initialise the weights for the two single matrices to get mu and log_sigma_square from the last layer
        all_weights = networks.fc_make_weights(all_weights, self.n_layers[-1], [self.n_output], add_b=False, id=1)
        all_weights = networks.fc_make_weights(all_weights, self.n_layers[-1], [self.n_output], add_b=False, id=2)

        return all_weights


class TemperaturePredictor(object):
    """
    Neural network to predict temperatures
    """

    def __init__(self, n_input, n_output, n_layers, name=None, nonlinearity=tf.nn.leaky_relu, temp_lim=5, lower_temp_lim=0.1):
        """
        Initialisation. This function should accept the external parameters, such as input and output sizes.

        :param n_input: dimensionality of input
        :param n_output: dimensionality of output
        :param n_layers: vector or list of dimensionality for each hidden layers, e.g. [100,50,50]
        :param name: name to assign to the Gaussian model
        :param nonlinearity: non-linearity to use (default is leaky relu)
        :param temp_lim: range to impose to the output temp to avoid divergence

        """

        self.n_input = n_input
        self.n_output = n_output
        self.n_layers = n_layers
        self.name = name
        self.bias_start = 0.0
        self.nonlinearity = nonlinearity
        self.temp_lim = temp_lim
        self.lower_temp_lim = lower_temp_lim

        # now we initialise the weights and set them as a global variable
        network_weights = self._create_weights()
        self.weights = network_weights

    def compute_temperature(self, x):
        """
        compute temperatures from inputs.

        :param x: batch of inputs [n_batch x dimensions]

        :return: inferred temperature.
        """

        # pass x through a fully connected network:
        hidden_post = networks.fc_network(x, self.weights, tf.shape(self.n_layers)[0], self.nonlinearity, id=0)

        # pass the previous layer into a linear model to get the temperature
        log_inv_temp = networks.fc_network(hidden_post, self.weights, 1, nonlinearity=None, add_b=False, id=1)
        lower_lit_lim = tfm.log(tf.divide(1.0, self.temp_lim))
        upper_lit_lim = tfm.log(tf.divide(1.0, self.lower_temp_lim))
        log_inv_temp = upper_lit_lim * (tf.nn.sigmoid(log_inv_temp / upper_lit_lim)) + lower_lit_lim
        return tf.divide(1.0, tfm.exp(log_inv_temp))


    def _create_weights(self):
        """
        Initialise weights.
        """

        # first, we initialise an empty ordered dictionary
        all_weights = collections.OrderedDict()

        # we make the weights for the fully connected network taking x:
        all_weights = networks.fc_make_weights(all_weights, self.n_input, self.n_layers, id=0)

        # We initialise the weights for the two single matrices to get mu and log_sigma_square from the last layer
        all_weights = networks.fc_make_weights(all_weights, self.n_layers[-1], [self.n_output], add_b=False, id=1)

        return all_weights
