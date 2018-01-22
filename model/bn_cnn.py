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

        to_next_layer = build_cnn_bn_pool_layer(image_batch, training, 1)[0]

        to_next_layer = build_cnn_bn_pool_layer(to_next_layer, training, 2)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 3, num_filter=64)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=64)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_bn_cnn_6(image_batch, training):
    """

    :param image_batch:
    :return:
    """
    scope_name = 'bn_cnn_6'

    with tf.variable_scope(scope_name):

        to_next_layer = build_cnn_bn_pool_layer(image_batch, training, 1)[0]

        to_next_layer = build_cnn_bn_pool_layer(to_next_layer, training, 2)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 3, num_filter=64)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=64)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 5, num_filter=128)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 6, num_filter=256)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_cnn_bn_pool_layer(image_batch,
                            training,
                            layer_number,
                            num_filter=32,
                            kernel_size=2,
                            strides=(1, 1),
                            bn_momentum=0.99,
                            pool_size=2,
                            pool_strides=2):
    """
    cnn -> bn -> max_pooling -> leaky_relu
    note:
      this layer dose not implemented as common way:(cnn -> bn -> activation -> pooling)
      because exchange the positions of pooling and activation can reduce computation costs (iif pooling is max_pooling)
    :param image_batch:
    :param training:
    :param layer_number:
    :param num_filter:
    :param kernel_size:
    :param strides:
    :param bn_momentum:
    :param pool_size:
    :param pool_strides:
    :return:
    """

    conv_name = 'conv_{0}'.format(layer_number)
    bn_name = 'bn_{0}'.format(layer_number)
    activation_name = 'activation_{0}'.format(layer_number)
    pool_name = 'pool_{0}'.format(layer_number)

    cnn_out = tf.layers.conv2d(
        inputs=image_batch,
        filters=num_filter,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        name=conv_name)

    bn_out = tf.layers.batch_normalization(
        cnn_out, momentum=bn_momentum, training=training, name=bn_name)

    pooling_out = tf.layers.max_pooling2d(
        inputs=bn_out,
        pool_size=pool_size,
        strides=pool_strides,
        padding='same',
        name=pool_name)

    activation_out = tf.nn.leaky_relu(pooling_out, name=activation_name)

    return activation_out, pooling_out, bn_out, cnn_out
