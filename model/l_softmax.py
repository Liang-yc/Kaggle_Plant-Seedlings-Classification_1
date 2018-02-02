#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from scipy.special import binom

def l_softmax(input, target, num_class, margin, training, name):
    """
    https://arxiv.org/abs/1612.02295
    https://github.com/jihunchoi/lsoftmax-pytorch/blob/master/lsoftmax.py
    :return:
    """

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=[input.shape[1], num_class],
                                 initializer=tf.contrib.layers.xavier_initializer)

        logits = tf.cond(training,
                         lambda: l_softmax_training(input, target, margin, weight),
                         lambda: tf.matmul(input, weight))

        return logits


def find_k(cos, divisor):
    # stop gradient
    acos = tf.acos(cos)
    k = tf.floor(acos / divisor).detach()
    return k


def l_softmax_training(input, target, margin, weight):

    divisor = tf.constant(np.pi / margin)
    coeffs = tf.constant(binom(margin, range(0, margin + 1, 2)))
    cos_exps = tf.constant(range(margin, -1, -2))
    sin_sq_exps = tf.constant(range(len(cos_exps)))
    signs = [1]
    for i in range(1, len(sin_sq_exps)):
        signs.append(signs[-1] * -1)
    signs = tf.constant(signs)


    # weight = tf.nn.l2_normalize(weight, dim=1, name='weight_l2_normalize')
    input_norm = tf.norm(input, ord=2, axis=1, keep_dims=True,
                         name='input_l2_norm')
    weight_norm = tf.norm(weight, ord=2, axis=1, keep_dims=True,
                          name='weight_l2_norm')

    logits = tf.matmul(input, weight)

    batch_index = tf.range(tf.shape(input)[0])
    # [[sample_0, sample_0_target_index], [sample_1, sample_1_target_index], [sample_2, sample_2_target_index]]
    logits_target_indices = tf.transpose(tf.stack([batch_index, target]))

    logits_target = tf.gather_nd(logits, logits_target_indices)
    weight_target_norm = weight_norm[:, target]

    norm_target_prod = weight_target_norm * input_norm
    # cos_target: (batch_size,)
    cos_target = logits_target / (norm_target_prod + 1e-10)
    sin_sq_target = 1 - cos_target ** 2

    cos_terms = tf.pow(tf.expand_dims(cos_target, 1), tf.expand_dims(cos_exps, 0))
    sin_sq_terms = tf.pow(tf.expand_dims(sin_sq_target, 1), tf.expand_dims(sin_sq_exps, 0))
    cosm_terms = (tf.expand_dims(signs, 0) * tf.expand_dims(coeffs, 0)
                  * cos_terms * sin_sq_terms)
    cosm = tf.reduce_sum(cosm_terms, 1)



def a_softmax():
    """
    https://arxiv.org/abs/1704.08063
    :return:
    """
    return None