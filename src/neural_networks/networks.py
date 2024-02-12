#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0


import tensorflow as tf
from neural_networks import layers


def fc_network(x_in, w, nl, nonlinearity, add_b=True, id=0):
    """
    run fully connected network

    :param x_in: input in the format [n_samples, dimensions_in]
    :param w: dictionary containing linear and additive weights (created with fc_make_weights)
    :param nl: number of layers
    :param nonlinearity: on-linearity to be used, e.g. tf.nn.relu
    :param add_b: boolean of whether to include additive weights b
    :param id: identifier number for the network (appended to the name of each weight)

    :return: output of the network in the format [n_samples, dimensions_out]
    """
    num_layers_1 = nl
    for i in range(num_layers_1):
        ni = i+1
        if add_b:
            x_in = layers.tf_fc_layer(x_in,w['W_h{}_to_h{}_ID{}'.format(ni-1,ni,id)],w['b_h{}_to_h{}_ID{}'.format(ni-1,ni,id)],nonlinearity)
        else:
            x_in = tf.matmul(x_in, w['W_h{}_to_h{}_ID{}'.format(ni-1,ni,id)])
    return x_in


def fc_make_weights(w_dict, n_in, layers_dimensions, add_b=True, id=0):
    """
    make weights for fully connected network

    :param w_dict: dictionary of weights to append the newly created ones to (if this is the first network, initialise it with 'w_dict={}')
    :param n_in: dimensionality of the input
    :param layers_dimensions: list or array of layers' dimensionalities [n_layers]. e.g. for a two layers network of 200 hidden units per layer this would be [200, 200]
    :param nonlinearity: on-linearity to be used, e.g. tf.nn.relu
    :param add_b: boolean of whether to include additive weights b
    :param id: identifier number for the network (appended to the name of each weight)

    :return: the weights' dictionary with the newly created weights appended to it
    """
    num_layers_1 = tf.shape(layers_dimensions)[0]
    layers_dimensions = tf.concat([tf.expand_dims(n_in,axis=0),layers_dimensions],axis=0)
    for i in range(num_layers_1):
        ni = i+1
        
        w_dict['W_h{}_to_h{}_ID{}'.format(ni-1,ni,id)] = layers.tf_fc_weights_w(layers_dimensions[ni-1],layers_dimensions[ni])
        if add_b==True: 
            w_dict['b_h{}_to_h{}_ID{}'.format(ni-1,ni,id)] = layers.tf_fc_weights_b(layers_dimensions[ni])
            
    return w_dict
    