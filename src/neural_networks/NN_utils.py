#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from tensorflow import math as tfm


def xavier_init(fan_in, fan_out, constant = 1):
    """
    Xavier initialisation for neural network layers' weights

    :param fan_in: dimensions of the input
    :param fan_out: dimensions of the output
    :param constant: scaling constant

    :return: initialised linear weights of the layer
    """
    low = -constant * tfm.sqrt(6.0 / (tf.cast(fan_in,dtype=tf.float32) + tf.cast(fan_out,dtype=tf.float32)))
    high = constant * tfm.sqrt(6.0 / (tf.cast(fan_in,dtype=tf.float32) + tf.cast(fan_out,dtype=tf.float32)))
    return tf.random.uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)
