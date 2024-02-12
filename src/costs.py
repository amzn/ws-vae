#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0


import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

SMALL_CONSTANT = 1e-5


def kl_normal(mu_1, log_sig_sq_1, mu_2, log_sig_sq_2):
    """
    Element-wise KL divergence between two normal distributions

    :param mu_1: mean of first distribution [n_batch x dimensions]
    :param log_sig_sq_1: log variance of first distribution [n_batch x dimensions]
    :param mu_2: mean of second distribution [n_batch x dimensions]
    :param log_sig_sq_2: log variance of second distribution [n_batch x dimensions]

    :return: element-wise kl divergence [n_batch x dimensions]
    """

    v_mean = mu_2  # 2
    aux_mean = mu_1  # 1
    v_log_sig_sq = log_sig_sq_2  # 2
    aux_log_sig_sq = log_sig_sq_1  # 1
    v_log_sig = tfm.log(tfm.sqrt(tfm.exp(v_log_sig_sq)))  # 2
    aux_log_sig = tfm.log(tfm.sqrt(tfm.exp(aux_log_sig_sq)))  # 1
    kl = v_log_sig - aux_log_sig + tf.divide(tfm.exp(aux_log_sig_sq) + tfm.square(aux_mean - v_mean),
                                             2.0 * tfm.exp(v_log_sig_sq)) - 0.5

    return kl


def reparameterisation_trick(mu, log_sig_sq):
    """
    Sample from Gaussian such that it stays differentiable

    :param mu: mean of distribution [n_batch x dimensions]
    :param log_sig_sq: log variance of distribution [n_batch x dimensions]

    :return: samples from distribution [n_batch x dimensions]
    """

    eps = tf.random.normal(tf.shape(mu), 0, 1., dtype=tf.float32)
    return tfm.add(mu, tfm.multiply(tfm.sqrt(tfm.exp(log_sig_sq)), eps))


def gaussian_log_likelihood(x, mu_x, log_sig_sq_x):
    """
    Element-wise Gaussian log likelihood

    :param x: data points [n_batch x dimensions]
    :param mu_x: means of Gaussians [n_batch x dimensions]
    :param log_sig_sq_x: log variances of Gaussians [n_batch x dimensions]

    :return: element-wise log likelihood
    """

    # -E_q(z|x) log(p(x|z))
    normalising_factor = - 0.5 * tfm.log(SMALL_CONSTANT + tfm.exp(log_sig_sq_x)) - 0.5 * np.log(2.0 * np.pi)
    square_diff_between_mu_and_x = tfm.square(mu_x - x)
    inside_exp = -0.5 * tfm.divide(square_diff_between_mu_and_x, SMALL_CONSTANT + tfm.exp(log_sig_sq_x))

    return normalising_factor + inside_exp


def weak_labels_reconstruction_likelihood(zyc, weak_labels, weak_label_decoder):
    """
    Weak labels reconstruction cost

    :param zyc: latent variables, comprising z and continuous label y_c concatenated [n_batch x n_z+1]
    :param weak_labels: weak labels [n_batch x n_lf]
    :param weak_label_decoder: decoder for temperatures of weak labels' distributions

    :return: VAE batch cost
    """
    y_c = tf.tile(tf.expand_dims(zyc[:, 0], axis=1), [1, tf.shape(weak_labels)[1]])
    mask = tf.cast(tfm.abs(weak_labels), tf.float32)
    temp = weak_label_decoder.compute_temperature(zyc) + SMALL_CONSTANT
    p_correct = tfm.log(tf.nn.sigmoid(tf.multiply(tf.divide(1.0, temp), tf.multiply(weak_labels, y_c))))
    p_correct = tfm.multiply(mask, p_correct)
    return tfm.reduce_mean(tfm.reduce_sum(p_correct, axis=1), axis=0)


def ws_vae_cost(x, weak_labels, encoder, decoder, weak_label_decoder, beta=1.0, gamma=1.0):
    """
    Cost function for WS-VAE

    :param x: inputs (e.g. pre-trained BERT encodings) [n_batch x dimensions]
    :param weak_labels: weak labels [n_batch x n_lf]
    :param encoder: encoder of the WS-VAE
    :param decoder: decoder of the WS-VAE
    :param weak_label_decoder: decoder for temperatures of weak labels' distributions
    :param beta: weight to give to the KL divergence (default=1 results in standard ELBO)
    :param gamma: weight to give to the feedback cost term (default=1)
    :param prior_bias: bias to give to prior distribution of labels (devault=0 => no bias)

    :return: WS-VAE batch cost (negative ELBO)
    """
    x = tf.cast(x, tf.float32)
    weak_labels = tf.cast(weak_labels, tf.float32)

    # compute moments of q(z, y_c|x, Lambda)
    mu_zyc, log_sig_sq_zyc = encoder.compute_moments(x)

    # define moments of prior p(z)p(y_c)
    mu_cz = tf.zeros([tf.shape(mu_zyc)[0], tf.shape(mu_zyc)[1]])
    log_sig_sq_cz = tf.zeros(tf.shape(mu_zyc))

    # sample from q(z, y_c|x, Lambda)
    zyc = reparameterisation_trick(mu_zyc, log_sig_sq_zyc)

    # compute moments of p(x|z, y_c)
    mu_x, log_sig_sq_x = decoder.compute_moments(zyc)

    # KL(q(z, y_c|x, Lambda)|p(z)p(y_c))
    kle = kl_normal(mu_zyc, log_sig_sq_zyc, mu_cz, log_sig_sq_cz)
    beta_v = beta*tf.ones([tf.shape(mu_zyc)[0], 1])
    beta_m = tf.concat([beta_v, 1.0*tf.ones([tf.shape(mu_zyc)[0], tf.shape(mu_zyc)[1] - 1])], axis=1)
    klc = tfm.reduce_sum(tf.multiply(beta_m,kle), 1)
    kl = tfm.reduce_mean(tf.cast(klc, tf.float32))

    # inputs reconstruction cost -E_q(z, y_c|x, Lambda) log(p(x|z, y_c))
    reconstr_loss = -tfm.reduce_sum(gaussian_log_likelihood(x, mu_x, log_sig_sq_x), 1)
    cost_r = tfm.reduce_mean(reconstr_loss)

    # weak labels reconstruction cost -E_q(z, y_c|x, Lambda) log(p(Lambda|z, y_c))
    cost_r_weak_labels = -weak_labels_reconstruction_likelihood(zyc, weak_labels, weak_label_decoder)

    return cost_r + kl + gamma * cost_r_weak_labels
