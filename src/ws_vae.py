#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0


import os
import numpy as np
import scipy
import tensorflow as tf
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, roc_auc_score
import training, constants
import var_logger
from neural_networks.gaussian_networks import GaussianModel, TemperaturePredictor


class WeakSupervisionVAE(object):
    """
    WS-VAE class
    """

    def __init__(self, example_data,
                 n_z = 10,
                 n_layers_encoder = [100],
                 n_layers_decoder = [100],
                 n_layers_weak_labelling_decoder = [20],
                 higher_temp_lim=5,
                 lower_temp_lim=0.1,
                 save_dir='src/neural_networks/saved_weights/weak_labelling_vae'):
        """
        initialisation

        :param example_data: dictionary containing sample of data, comprising at least features [n_batch, n_dim] and weak labels [n_batch, n_weak_labels]
        :param n_z: number of latent dimensions
        :param n_layers_encoder: number of hidden units per layer for encoder neural network (default=[100] single layer of 100 units)
        :param n_layers_decoder: number of hidden units per layer for decoder neural network (default=[100] single layer of 100 units)
        :param n_layers_weak_labelling_decoder: number of hidden units per layer for weak labels decoder neural network (default=[20] single layer of 20 units)
        :param higher_temp_lim: higher limit for learned weak labels likelihoods' temperatures
        :param lower_temp_lim: lower limit for learned weak labels likelihoods' temperatures
        :param save_dir: directory in which to save the weights of the WS-VAE
        """

        # inputs to class global variables
        self.data = example_data
        self.inputs = example_data.features
        self.weak_labels = example_data.weak_labels
        self.save_dir = save_dir

        # class global classes
        self.encoder = GaussianModel(np.shape(self.inputs)[1], n_z, n_layers_encoder, name=constants.ENCODER)
        self.decoder = GaussianModel(n_z, np.shape(self.inputs)[1], n_layers_decoder, name=constants.DECODER, constrain=True)
        self.weak_label_decoder = TemperaturePredictor(n_z, np.shape(self.weak_labels)[1], n_layers_weak_labelling_decoder, name=constants.WEAK_LABELLING_DECODER, temp_lim=higher_temp_lim, lower_temp_lim=lower_temp_lim)

    def fit(self, dataset_train=None,
            dataset_valid=None,
            beta=1.0,
            gamma=1.0,
            num_iterations=10001,
            initial_training_rate=0.001,
            batch_size=32,
            warmup_start=0,
            warmup_end=0,
            report_interval=250,
            print_values=True
            ):
        """
        training function for the WS-VAE. It trains the model and, if validation data is given, makes vectors of validation cost and kappa score.

        :param dataset_train: class containing training data, comprising at least features [n_train, n_dim] and weak labels [n_train, n_weak_labels]
        :param dataset_valid: class containing validation data, comprising at least features [n_valid, n_dim], weak labels [n_valid, n_weak_labels] and ground-truth labels [n_valid]
        :param beta: weight to give to the KL divergence (default=1 results in standard ELBO)
        :param gamma: weight to give to the feedback cost term (default=1)
        :param num_iterations: number of iterations
        :param initial_training_rate: initial training rate for ADAM optimiser
        :param batch_size: training batch size
        :param warmup_start: iteration at which to start KL warm-up
        :param warmup_end: iteration at which to end KL warm-up (default=0 means no warm-up)
        :param report_interval: interval in iteration at which to compute validation/training performance and optionally print values
        :param print_values: whether to periodically print info while training the WS-VAE

        :return: 1D array of cost (validation if val set is provided, training otherwise) every 'report_interval' iterations
        """

        # use data from class definition if training data not given
        if dataset_train is None:
            dataset_train = self.data

        # optimisation parameters
        optimisation_parameters = dict(
            print_values=print_values,
            num_iterations=num_iterations,
            initial_training_rate=initial_training_rate,
            batch_size=batch_size,
            warmup_start=warmup_start,
            warmup_end=warmup_end,
            report_interval=report_interval
        )

        inputs = np.asarray(dataset_train.features).astype('float32')
        weak_labels = var_logger.switch_minus_ones_and_zeros(np.asarray(dataset_train.weak_labels).astype('float32'))

        if dataset_valid is not None:

            inputs_val = np.asarray(dataset_valid.features).astype('float32')
            weak_labels_val = var_logger.switch_minus_ones_and_zeros(np.asarray(dataset_valid.weak_labels).astype('float32'))
            true_labels_val = var_logger.switch_minus_ones_and_zeros(np.asarray(dataset_valid.labels).astype('float32'))


            cost_plot = training.train_ws_vae(self.encoder, self.decoder, self.weak_label_decoder, inputs, weak_labels,
                                                          optimisation_parameters, inputs_val, weak_labels_val, true_labels_val,
                                                          beta=beta, gamma=gamma, save_dir=self.save_dir)

        else:
            cost_plot = training.train_ws_vae(self.encoder, self.decoder, self.weak_label_decoder, inputs, weak_labels,
                                  optimisation_parameters, beta=beta, gamma=gamma, save_dir=self.save_dir)

        return cost_plot

    def load(self):
        """
        Load pre-trained parameters (from path in class variable save_dir)
        """
        if os.path.exists(os.path.join(self.save_dir, 'enc.mat')):
            save_name_encoder = self.save_dir + '/enc.mat'
            save_name_decoder = self.save_dir + '/dec.mat'
            save_name_decoder_wl = self.save_dir + '/dec_wl.mat'
            var_logger.restore_dict(save_name_encoder, self.encoder.weights)
            var_logger.restore_dict(save_name_decoder, self.decoder.weights)
            var_logger.restore_dict(save_name_decoder_wl, self.weak_label_decoder.weights)
        else:
            print('weights in {} not found. Please train the model to get weights before loading')

    def encode(self, new_inputs):
        """
        encode inputs and weak labels to continuous label y_c and latent variable z (concatenated).

        :param new_inputs: inputs to encode [n_batch, n_dim]

        :return: mean [n_batch, n_z+1] and log variance [n_batch, n_z+1] of continuous label y_c and latent variable z
        """
        new_inputs = tf.cast(new_inputs, tf.float32)
        new_inputs_lambda = new_inputs
        mu_z, log_sig_sq_z = self.encoder.compute_moments(new_inputs_lambda)
        return mu_z.numpy(), log_sig_sq_z.numpy()

    def infer_continuous_label(self, new_inputs):
        """
        infer distribution of continuous label y_c from features array.

        :param new_inputs: inputs to encode [n_batch, n_dim]

        :return: mean [n_batch] and log variance [n_batch] of continuous label y_c
        """
        mu_z, log_sig_sq_z = self.encode(new_inputs)
        return mu_z[:, 0], log_sig_sq_z[:, 0]

    def predict_continuous_label(self, data_new):
        """
        predict continuous labels y_c from data class.

        :param data_new: dictionary containing data to infer soft labels from, comprising at least features [n_batch, n_dim]

        :return: mean [n_batch] and log variance [n_batch] of continuous label y_c
        """
        new_inputs = np.asarray(data_new.features).astype('float32')
        mu_yc, lss_yc = self.infer_continuous_label(new_inputs)
        return mu_yc, lss_yc

    def predict_proba(self, data_new):
        """
        predict soft labels.

        :param data_new: dictionary containing data to infer soft labels from, comprising at least features [n_batch, n_dim]

        :return: soft labels [n_batch x 2] => [p(y=0 | x,Lambda), p(y=1 | x,Lambda)]
        """
        new_inputs = np.asarray(data_new.features).astype('float32')
        mu_yc, lss_yc = self.infer_continuous_label(new_inputs)
        x = -np.divide(mu_yc, np.sqrt(np.exp(lss_yc)))
        p_l0 = 0.5*(scipy.special.erf(x)+1.0)
        return np.concatenate((np.expand_dims(p_l0, axis=1), np.expand_dims(1.0-p_l0, axis=1)), axis=1)

    def predict(self, data_new):
        """
        predict hard labels.

        :param data_new: dictionary containing data to infer soft labels from, comprising at least features [n_batch, n_dim]

        :return: inferred labels (binary) [n_batch]
        """
        soft_labels = self.predict_proba(data_new)
        return np.argmax(soft_labels, axis=1)

    def test(self, data_new, metric_fn):
        """
        test WS-VAE and return classification metric.

        :param data_new: dictionary containing data to infer soft labels from, comprising at least features [n_batch, n_dim] and true labels [n_batch]
        :param metric_fn: type of metric to compute, one of 'acc', 'kappa' and 'matrix'

        :return: classification metric.
        """
        hard_labels = self.predict(data_new)
        if metric_fn == 'kappa':
            return cohen_kappa_score(data_new.labels.astype(int), hard_labels.astype(int))
        elif metric_fn == 'matrix':
            return confusion_matrix(data_new.labels.astype(int), hard_labels.astype(int))
        elif metric_fn == 'f1_binary':
            return f1_score(data_new.labels.astype(int), hard_labels.astype(int))
        elif metric_fn == 'acc':
            return accuracy_score(data_new.labels.astype(int), hard_labels.astype(int))
        elif metric_fn == 'auc':
            soft_labels = self.predict_proba(data_new)
            return roc_auc_score(np.asarray(data_new.labels).astype('float32'), soft_labels[:,1])
        else:
            print('metric {} not yet implemented'.format(metric_fn))


