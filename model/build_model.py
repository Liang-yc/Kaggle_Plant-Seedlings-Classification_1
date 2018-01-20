#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

def add_gradients_summaries(grads_and_vars):
    """Add summaries to gradients.
    Args:
      grads_and_vars: A list of gradient to variable pairs (tuples).
    Returns:
      The list of created summaries.
    """
    summaries = []
    for grad, var in grads_and_vars:
        if grad is not None:
            if isinstance(grad, ops.IndexedSlices):
                grad_values = grad.values
            else:
                grad_values = grad
            summaries.append(
                tf.summary.histogram(var.op.name + '_gradient', grad_values))
            summaries.append(
                tf.summary.scalar(var.op.name + '_gradient_norm',
                               clip_ops.global_norm([grad_values])))
        else:
            logging.info('Var %s has no gradient', var.op.name)

    return summaries

def build_simple_cnn(image_batch):
    """
    conv -- pool -- conv -- pool - flatten
    :param image_batch:
    :return:
    """
    slim = tf.contrib.slim
    scope_name = "simple_cnn"

    with tf.variable_scope(scope_name):

        to_next_layer = tf.layers.conv2d(
            inputs=image_batch,
            filters=32,
            kernel_size=2,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            name='conv_1'
        )

        to_next_layer = tf.layers.max_pooling2d(
            inputs=to_next_layer,
            pool_size=2,
            strides=1,
            padding='same',
            name='pool_1'
        )

        to_next_layer = tf.layers.conv2d(
            inputs=image_batch,
            filters=32,
            kernel_size=2,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            name='conv_2'
        )

        to_next_layer = tf.layers.max_pooling2d(
            inputs=to_next_layer,
            pool_size=2,
            strides=1,
            padding='same',
            name='pool_2'
        )

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_classifier(image_batch, num_class):
    """
    cnn -- fc -- softmax
    :param image_batch:
    :param num_class:
    :return: unscaled logits, scaled logits, trainable var
    """
    scope_name = "plant_seedings_classifier"
    with tf.variable_scope(scope_name):
        flatten = build_simple_cnn(image_batch)

        linear = tf.layers.dense(flatten, num_class, name='fc')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)

def build_loss(sparse_labels, unscaled_logits):
    reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())

    ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
        sparse_labels, unscaled_logits, scope='softmax_cross_entropy_loss'))

    return reg_loss + ce

def build_train_op(loss, trainable, global_step):
    """
    :param loss:
    :param trainable:
    :return: optimizer wrt trainable
    """
    optimizer = tf.train.AdamOptimizer()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if update_ops:
        with tf.control_dependencies([tf.group(*update_ops)]):
            grad_var = optimizer.compute_gradients(loss, var_list=trainable)
            add_gradients_summaries(grad_var)
            return optimizer.apply_gradients(grad_var, global_step=global_step)
    else:
        grad_var = optimizer.compute_gradients(loss, var_list=trainable)
        add_gradients_summaries(grad_var)
        return optimizer.apply_gradients(grad_var, global_step=global_step)