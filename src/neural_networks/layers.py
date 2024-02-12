#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from neural_networks import NN_utils


def tf_fc_layer(x_in, w, b, nonlinearity):
    """
    run fully connected layer

    :param x_in: input features [n_samples, n_in]
    :param w: linear weights [n_in, n_out]
    :param b: linear weights [n_in, n_out]
    :param nonlinearity: non-linearity to be used, e.g. tf.nn.relu

    :return: output of the layer in the format [n_samples, n_out]
    """
    y_out = tf.add(tf.matmul(x_in, w), b)
    if nonlinearity:
        y_out = nonlinearity(y_out)
    return y_out

def tf_fc_weights_w(n_h_in,n_h_out):
    """
    initialise linear weights for fully connected layer

    :param n_h_in: number dimensions input
    :param n_h_out: number dimensions out

    :return: initialised linear weights in the format [n_h_in,n_h_out]
    """
    w = tf.Variable(NN_utils.xavier_init(n_h_in,n_h_out), dtype=tf.float32)
    return w

def tf_fc_weights_b(n_h_out):
    """
    initialise additive weights for fully connected layer

    :param n_h_out: number dimensions out (dimensionality of additive weights)

    :return: initialised additive weights in the format [n_h_out]
    """
    b = tf.Variable(tf.zeros(n_h_out, dtype=tf.float32))
    return b
