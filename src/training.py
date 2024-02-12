#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: Apache-2.0


import numpy as np
import tensorflow as tf
import scipy
import costs, var_logger, constants
from sklearn.metrics import f1_score


def sequential_indexer(data_length, batch_size, iteration, permuted_indices=None):
    """
    Sequential indexer to make indices for training batches.

    :param data_length: size of full training data.
    :param batch_size: batch size.
    :param iteration: current iteration.
    :param permuted_indices: permuted list or array of indeces, in case randomisation of the set is needed.

    :return: indices for the batch.
    """

    if permuted_indices is None:
        indices = np.arange(data_length)
    else:
        indices = permuted_indices

    max_iteration = np.floor(np.divide(data_length, batch_size))

    while iteration > max_iteration:
        iteration = iteration - max_iteration

    sub = np.multiply(iteration, batch_size)

    if sub < data_length - batch_size:
        return indices[sub.astype(int):(sub + batch_size).astype(int)]
    else:
        reminder = batch_size - data_length + sub
        return np.concatenate((indices[sub.astype(int):], indices[:reminder.astype(int)]), axis=0)


def train_ws_vae(encoder, decoder, weak_label_decoder, inputs, weak_labels, params, inputs_val=None,
                                    weak_labels_val=None, true_labels_val=None, beta=1.0, gamma=1.0,
                                    save_dir='ws_vae/neural_networks/saved_weights/weak_labelling_vae'):
    """
    Function to train a WS-VAE

    :param encoder: encoder of the WS-VAE
    :param decoder: decoder of the WS-VAE
    :param weak_label_decoder: decoder for temperatures of weak labels' distributions
    :param inputs: training inputs (e.g. pre-trained BERT encodings) [n_samples x dimensions]
    :param weak_labels: training weak labels [n_samples x n_lf]
    :param inputs_val: validation inputs (e.g. pre-trained BERT encodings) [n_samples x dimensions]
    :param weak_labels_val: validation weak labels [n_samples x n_lf]
    :param true_labels_val: validation ground-truth labels [n_samples]
    :param beta: weight to give to the KL divergence (default=1 results in standard ELBO)
    :param gamma: weight to give to the feedback cost term (default=1)
    :param save_dir: path to which to save the trained weights of the WS-VAE

    :return: vector of validation WS-VAE cost and vector of validation kappa score as functions of iteration
    """

    # Initialise the arrays in which to save the cost plots
    cost_plot = np.zeros(int(np.round(params[constants.NUM_ITERATIONS] / params[constants.REPORT_INTERVAL]) + 1))

    # ADAM optimiser initialisation
    optimizer = tf.keras.optimizers.Adam(params[constants.INITIAL_TRAINING_RATE], clipnorm=0.1)

    # initialise beta parameter for KL divergence to use in warmup
    if params[constants.WARMUP_END] - params[constants.WARMUP_START] > 0:
        beta_i = 0.0
    else:
        beta_i = beta

    # Start iterations
    ni = -1
    best_f1 = 0.0
    for i in range(params[constants.NUM_ITERATIONS]):

        # warm-up step
        if i > params[constants.WARMUP_START] and i <= params[constants.WARMUP_END]:
            beta_i = beta_i + np.divide(beta, params[constants.WARMUP_END] - params[constants.WARMUP_START])

        # random indices for the next batch
        # next_indices = np.random.random_integers(np.shape(inputs)[0], size=(params[constants.BATCH_SIZE])) - 1
        next_indices = sequential_indexer(np.shape(inputs)[0], params[constants.BATCH_SIZE], i)

        # take inputs and weak labels for the next batch
        inputs_batch = inputs[next_indices, :]
        weak_labels_batch = weak_labels[next_indices, :]
        # Optimisation Step
        cost = lambda: costs.ws_vae_cost(inputs_batch, weak_labels_batch, encoder, decoder,
                                                     weak_label_decoder, beta=beta_i, gamma=gamma)
        optimizer.minimize(cost, var_list=[encoder.weights, decoder.weights, weak_label_decoder.weights])

        # If validation available, periodically compute cost and f1-score with true labels over validation set
        if i % params[constants.REPORT_INTERVAL] == 0:
            ni = ni + 1

            if inputs_val is not None:

                # overall validation cost
                cost = costs.ws_vae_cost(inputs_val, weak_labels_val, encoder, decoder, weak_label_decoder,
                                         beta=beta_i, gamma=gamma)

                # infer continuous labels
                inputs_val = tf.cast(inputs_val, tf.float32)
                mu_yc, lss_yc = encoder.compute_moments(inputs_val)
                mu_yc = mu_yc[:,0].numpy()
                lss_yc = lss_yc[:, 0].numpy()
                # compute hard labels
                p_arg = -np.divide(mu_yc, np.sqrt(np.exp(lss_yc)))
                p_l0 = 0.5 * (scipy.special.erf(p_arg) + 1.0)
                y_val = np.argmax(np.concatenate((np.expand_dims(p_l0, axis=1), np.expand_dims(1.0-p_l0, axis=1)), axis=1), axis=1)
                # compute f1 over validation set for early stopping
                f1 = f1_score(var_logger.switch_minus_ones_and_zeros(true_labels_val.astype(int)), y_val.astype(int))

                # put cost value in the plot
                cost_plot[ni] = cost.numpy()

                # save weights if f1 is the best so far
                if f1 > best_f1:
                    save_name_encoder = save_dir + '/enc.mat'
                    save_name_decoder = save_dir + '/dec.mat'
                    save_name_decoder_wl = save_dir + '/dec_wl.mat'
                    var_logger.save_dict(save_name_encoder, encoder.weights)
                    var_logger.save_dict(save_name_decoder, decoder.weights)
                    var_logger.save_dict(save_name_decoder_wl, weak_label_decoder.weights)
                    best_f1 = f1

            else:

                # batch training cost
                cost = costs.ws_vae_cost(inputs_batch, weak_labels_batch, encoder, decoder, weak_label_decoder,
                                         beta=beta_i, gamma=gamma)
                cost_plot[ni] = cost.numpy()

            # print out values if needed
            if params[constants.PRINT_VALUES]:
                if inputs_val is not None:
                    print('Training satisfaction VAE model, Iteration:', i, '/', params[constants.NUM_ITERATIONS],
                         ', validation cost value:', cost.numpy(), ', validation f1-score:', f1)
                else:
                    print('Training satisfaction VAE model, Iteration:', i, '/', params[constants.NUM_ITERATIONS],
                          ', training batch cost value:', cost.numpy())

            # stop if we get a numerical evaluation error
            if np.isnan(cost):
                print('NaN Error! Iteration:', i, '/', params[constants.NUM_ITERATIONS])
                break

    if best_f1 > 0.0:
        var_logger.restore_dict(save_name_encoder, encoder.weights)
        var_logger.restore_dict(save_name_decoder, decoder.weights)
        var_logger.restore_dict(save_name_decoder_wl, weak_label_decoder.weights)

    return cost_plot
