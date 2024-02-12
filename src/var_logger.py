#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0


import os
import scipy.io as sio
import tensorflow as tf
import json
import numpy as np


def switch_minus_ones_and_zeros(x):
    """
    switch -1 and 0 in given array. This is because conventionally weak labels have -1 = abstain and 0 = negative,
    while WS-VAE modelling requires the opposite.

    :param x: input array. Must be an array with 0, 1 and -1 entries only.

    :return: array with -1 and 0 exchanged.
    """
    y = np.zeros(np.shape(x))
    y[x==0.0] = -1.0
    y[x==1.0] = 1.0
    return y


def save_dict(save_dir, dict_in):
    """
    Save tensorflow weights as a dictionary in a .mat file

    :param save_dir: directory and file name (including extension) in which to save the weights
    :param dict_in: tensorflow2 dictionary of weights
    """
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    dict_new = {}
    for key, value in dict_in.items():
        dict_new[key] = value.numpy()
    sio.savemat(save_dir, dict_new)


def restore_dict(load_dir, init_dict):
    """
    Load weights from a .mat file and return them as a dictionary tensorflow tensors

    :param load_dir: directory and file name (including extension) from which to load the weights
    :param init_dict: tensorflow2 dictionary of initialised weights (before loading the pre-trained values)
    """
    pretrained_weights = sio.loadmat(load_dir)
    for key, value in init_dict.items():
        init_dict[key] = tf.Variable(tf.constant(pretrained_weights[key]), dtype=tf.float32)


def load_json(load_path):
    """
    load data from json file

    :param load_path: path including filename of json with data

    :return: data dictionary
    """
    f = open(load_path, )
    data = json.load(f)
    f.close()
    return data


def load_wrench_data(load_path):
    """
    load data from json file in the format of the data sets provided as part of the Wrench benchmark framework (https://github.com/JieyuZ2/wrench)

    :param load_path: path including filename of json with data

    :return: data set class in the format accepted by WS-VAE (which is the same as the class format used in Wrench)
    """
    data = load_json(load_path)
    for i in range(len(data)):
        data_point_i = data[list(data.keys())[i]]
        features_i = np.expand_dims(data_point_i['data']['feature'], axis=0)
        weak_labels_i = np.expand_dims(data_point_i['weak_labels'], axis=0)
        labels_i = np.expand_dims(data_point_i['label'], axis=0)
        if i==0:
            features = features_i
            weak_labels = weak_labels_i
            labels = labels_i
        else:
            features = np.concatenate((features, features_i), axis=0)
            weak_labels = np.concatenate((weak_labels, weak_labels_i), axis=0)
            labels = np.concatenate((labels, labels_i), axis=0)
    return WeakSupervisionDataset(features, weak_labels, labels)


class WeakSupervisionDataset(object):
    """
   data set class for WS
    """

    def __init__(self, features, weak_labels=None, labels=None):
        """
            load data from json file

            :param features: 2D array of features [n_samples, n_features]
            :param weak_labels: 2D array of weak labels [n_samples, n_weak_labels]
            :param labels: 1D array of ground-truth labels [n_samples]
            """
        # inputs to class global variables
        self.features = features
        self.weak_labels = weak_labels
        self.labels = labels