#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


def build_bn_cnn_4(image_batch, training):
    """

    :param image_batch:
    :return:
    """
    scope_name = 'bn_cnn_4'

    with tf.variable_scope(scope_name):

        to_next_layer = build_cnn_layer(image_batch, training, 1)

        to_next_layer = build_cnn_layer(to_next_layer, training, 2)

        to_next_layer = build_cnn_layer(
            to_next_layer, training, 3, num_filter=64)

        to_next_layer = build_cnn_layer(
            to_next_layer, training, 4, num_filter=64)

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_cnn_layer(image_batch,
                    training,
                    layer_number,
                    num_filter=32,
                    kernel_size=2,
                    strides=(1, 1),
                    bn_momentum=0.99,
                    pool_size=2,
                    pool_strides=2):

    conv_name = 'conv_{0}'.format(layer_number)
    bn_name = 'bn_{0}'.format(layer_number)
    activation_name = 'activation_{0}'.format(layer_number)
    pool_name = 'pool_{0}'.format(layer_number)

    to_next_layer = tf.layers.conv2d(
        inputs=image_batch,
        filters=num_filter,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        name=conv_name)

    to_next_layer = tf.layers.batch_normalization(
        to_next_layer, momentum=bn_momentum, training=training, name=bn_name)

    to_next_layer = tf.nn.leaky_relu(to_next_layer, name=activation_name)

    to_next_layer = tf.layers.max_pooling2d(
        inputs=to_next_layer,
        pool_size=pool_size,
        strides=pool_strides,
        padding='same',
        name=pool_name)

    return to_next_layer
