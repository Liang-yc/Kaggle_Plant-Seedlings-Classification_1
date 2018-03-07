#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import horovod.tensorflow as hvd

import model.bn_cnn
import model.l_softmax
import model.resnet_v2
import model.inception_resnet_v2

slim = tf.contrib.slim

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
    scope_name = "simple_cnn"

    with tf.variable_scope(scope_name):

        to_next_layer = tf.layers.conv2d(
            inputs=image_batch,
            filters=32,
            kernel_size=2,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            name='conv_1')

        to_next_layer = tf.layers.max_pooling2d(
            inputs=to_next_layer,
            pool_size=2,
            strides=1,
            padding='same',
            name='pool_1')

        to_next_layer = tf.layers.conv2d(
            inputs=to_next_layer,
            filters=32,
            kernel_size=2,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            name='conv_2')

        to_next_layer = tf.layers.max_pooling2d(
            inputs=to_next_layer,
            pool_size=2,
            strides=1,
            padding='same',
            name='pool_2')

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


def build_cnn_4_classifier(image_batch, num_class, training):
    """

    :param image_batch:
    :param num_class:
    :param training:
    :return:
    """

    scope_name = "plant_seedings_cnn_4_classifier"

    with tf.variable_scope(scope_name):
        flatten = model.bn_cnn.build_bn_cnn_4(image_batch, training)

        linear = tf.layers.dense(flatten, num_class, name='fc')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)


def build_cnn_6_classifier(image_batch, num_class, training):
    """

    :param image_batch:
    :param num_class:
    :param training:
    :return:
    """

    scope_name = "plant_seedings_cnn_6_classifier"

    with tf.variable_scope(scope_name):
        flatten = model.bn_cnn.build_bn_cnn_6(image_batch, training)

        linear = tf.layers.dense(flatten, num_class, name='fc')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)


def build_cnn_8_classifier(image_batch, num_class, training):
    """

    :param image_batch:
    :param num_class:
    :param training:
    :return:
    """

    scope_name = "plant_seedings_cnn_8_classifier"

    with tf.variable_scope(scope_name):
        flatten = model.bn_cnn.build_bn_cnn_8(image_batch, training)

        linear = tf.layers.dense(flatten, num_class, name='fc')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)


def build_cnn_8_crelu_classifier(image_batch, num_class, training):
    """

    :param image_batch:
    :param num_class:
    :param training:
    :return:
    """

    scope_name = "plant_seedings_cnn_8_crelu_classifier"

    with tf.variable_scope(scope_name):
        flatten = model.bn_cnn.build_bn_cnn_8_crelu(image_batch, training)

        linear = tf.layers.dense(flatten, num_class, name='fc')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)


def build_cnn_8_crelu_classifier_with_lsoftmax(image_batch, target, num_class, lambda_decay, training):

    scope_name = "plant_seedings_cnn_8_crelu_classifier_with_lsoftmax"

    with tf.variable_scope(scope_name):
        flatten = model.bn_cnn.build_bn_cnn_8_crelu(image_batch, training)

        linear = model.l_softmax.l_softmax(flatten, target, num_class, 4, lambda_decay, training, 'l_softmax')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)


def build_cnn_12_classifier_with_lsoftmax(image_batch, target, num_class, lambda_decay, training):

    scope_name = "plant_seedings_cnn_12_classifier_with_lsoftmax"

    with tf.variable_scope(scope_name):
        flatten = model.bn_cnn.build_bn_cnn_12(image_batch, training)

        linear = model.l_softmax.l_softmax(flatten, target, num_class, 4,
            lambda_decay, training, 'l_softmax')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)


def build_cnn_8_crelu_classifier_with_dropout(image_batch, num_class, training):
    """

    :param image_batch:
    :param num_class:
    :param training:
    :return:
    """

    scope_name = "plant_seedings_cnn_8_crelu_classifier_with_dropout"

    with tf.variable_scope(scope_name):
        flatten = model.bn_cnn.build_bn_cnn_8_crelu_with_dropout(image_batch, training)

        linear = tf.layers.dense(flatten, num_class, name='fc')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)


def build_cnn_8_crelu_deformable_classifier(image_batch, num_class, training):
    """

    :param image_batch:
    :param num_class:
    :param training:
    :return:
    """

    scope_name = "plant_seedings_cnn_8_crelu_deformable_classifier"

    with tf.variable_scope(scope_name):
        flatten = model.bn_cnn.build_bn_cnn_8_crelu_deformable(
            image_batch, training)

        linear = tf.layers.dense(flatten, num_class, name='fc')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)


def build_cnn_6_with_skip_connection_classifier(image_batch, num_class,
                                                training):
    """

    :param image_batch:
    :param num_class:
    :param training:
    :return:
    """

    scope_name = "plant_seedings_cnn_6_with_skip_connection_classifier"

    with tf.variable_scope(scope_name):
        flatten = model.bn_cnn.build_bn_cnn_6_with_skip_connection(
            image_batch, training)

        linear = tf.layers.dense(flatten, num_class, name='fc')

        logits = tf.nn.softmax(linear, name='softmax')

    return linear, logits, tf.trainable_variables(scope_name)


def build_resnet_v2_50(image_batch, target, num_class, lambda_decay, training):
    with slim.arg_scope(model.resnet_v2.resnet_arg_scope(batch_norm_decay=0.9)):
        _, end_points = model.resnet_v2.resnet_v2_50(image_batch, 1001, is_training=training)
        flatten = tf.layers.flatten(end_points['global_pool'])

    scope_name = "plant_seedings_build_resnet_v2_50_with_lsoftmax"
    with tf.variable_scope(scope_name):
        linear = model.l_softmax.l_softmax(flatten, target, num_class, 4,
            lambda_decay, training, 'l_softmax')

        logits = tf.nn.softmax(linear, name='softmax')
    return linear, logits, None


def buiid_inception_resnet_v2(image_batch, target, num_class, lambda_decay, training):
    with slim.arg_scope(model.inception_resnet_v2.inception_resnet_v2_arg_scope(batch_norm_decay=0.9)):
        _, end_points = model.inception_resnet_v2.inception_resnet_v2(image_batch, 1001, training)
        flatten = tf.layers.flatten(end_points['global_pool'])

    scope_name = "plant_seedings_build_inception_resnet_v2_with_lsoftmax"
    with tf.variable_scope(scope_name):
        linear = model.l_softmax.l_softmax(flatten, target, num_class, 4,
            lambda_decay, training, 'l_softmax')

        logits = tf.nn.softmax(linear, name='softmax')
    return linear, logits, None


def build_loss(sparse_labels, unscaled_logits):
    reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())

    ce = tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(
            sparse_labels, unscaled_logits, scope='softmax_cross_entropy_loss'))

    return reg_loss + ce


def build_train_op(loss, trainable, global_step):
    """
    :param loss:
    :param trainable:
    :return: optimizer wrt trainable
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001 * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)

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


def build_test_time_data_augmentation(x):
    """
    test time data augmentation
    input batch, output batch * 8
    x = [batch, height, width, channel]
    """
    x_rot_90 = tf.contrib.image.rotate(x, 90)
    x_rot_180 = tf.contrib.image.rotate(x, 180)
    x_rot_270 = tf.contrib.image.rotate(x, 270)

    x_flip = tf.reverse(x, [2])
    x_flip_rot_90 = tf.contrib.image.rotate(x_flip, 90)
    x_flip_rot_180 = tf.contrib.image.rotate(x_flip, 180)
    x_flip_rot_270 = tf.contrib.image.rotate(x_flip, 270)

    x = tf.concat([x, x_rot_90, x_rot_180, x_rot_270, x_flip, x_flip_rot_90,
        x_flip_rot_180, x_flip_rot_270], axis=0)

    return x


def build_test_time_vote(logits):
    """

    """
    logits = tf.one_hot(tf.argmax(logits, axis=1), depth=logits.shape[1])

    [logits, logits_rot_90, logits_rot_180, logits_rot_270, logits_flip, logits_flip_rot_90,
        logits_flip_rot_180, logits_flip_rot_270] = tf.split(logits, 8)

    logits = logits + logits_rot_90 + logits_rot_180 + logits_rot_270 + logits_flip + logits_flip_rot_90 + logits_flip_rot_180 + logits_flip_rot_270

    return logits

def restore_pretrained_resnet_v2_50(session):
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
        include=["resnet_v2_50"])
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        "pretrained_model/resnet_v2_50.ckpt", variables_to_restore, ignore_missing_vars=True)
    init_fn(session)
