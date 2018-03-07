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


class ModelBase():
    def __init__(self):
        return


    def build_model(self, image_batch, target, num_class, lambda_decay, training):
        raise NotImplementedError()
        return


    def build_train_op(self, loss, global_step, trainable=None,
                   optimizer=tf.train.AdamOptimizer):
        optimizer = optimizer(learning_rate=0.001 * hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if update_ops:
            with tf.control_dependencies([tf.group(*update_ops)]):
                grad_var = optimizer.compute_gradients(loss, var_list=trainable)
                add_gradients_summaries(grad_var)
                return optimizer.apply_gradients(grad_var,
                                                 global_step=global_step)
        else:
            grad_var = optimizer.compute_gradients(loss, var_list=trainable)
            add_gradients_summaries(grad_var)
            return optimizer.apply_gradients(grad_var, global_step=global_step)
        return


    def build_loss(self, sparse_labels, unscaled_logits):
        reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())

        ce = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(
                sparse_labels, unscaled_logits,
                scope='softmax_cross_entropy_loss'))

        return reg_loss + ce


    def load_pretrained_weight(self, file_name, session):
        return


    def restore_weight(self, file_name, session):
        return

    def load_pretrained_scope_weight(self, file_name, include_scope, session):
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=[include_scope])
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            file_name, variables_to_restore,
            ignore_missing_vars=True)
        init_fn(session)


class Cnn8CreluLsoftmax(ModelBase):
    def __init__(self):
        ModelBase()
        return


    def build_model(self, image_batch, target, num_class, lambda_decay, training):
        scope_name = "plant_seedings_cnn_8_crelu_classifier_with_lsoftmax"

        with tf.variable_scope(scope_name):
            flatten = model.bn_cnn.build_bn_cnn_8_crelu(image_batch, training)

            linear = model.l_softmax.l_softmax(flatten, target, num_class, 4,
                                               lambda_decay, training,
                                               'l_softmax')

            logits = tf.nn.softmax(linear, name='softmax')

        return linear, logits, tf.trainable_variables(scope_name)


class Resnet50V2(ModelBase):
    def __init__(self):
        ModelBase()
        return

    def build_model(self, image_batch, target, num_class, lambda_decay,
                    training):
        with slim.arg_scope(
                model.resnet_v2.resnet_arg_scope(batch_norm_decay=0.9)):
            _, end_points = model.resnet_v2.resnet_v2_50(image_batch, 1001,
                                                         is_training=training)
            flatten = tf.layers.flatten(end_points['global_pool'])

        scope_name = "plant_seedings_build_resnet_v2_50_with_lsoftmax"
        with tf.variable_scope(scope_name):
            linear = model.l_softmax.l_softmax(flatten, target, num_class, 4,
                                               lambda_decay, training,
                                               'l_softmax')

            logits = tf.nn.softmax(linear, name='softmax')
        return linear, logits, None

    def load_pretrained_weight(self, file_name, session):
        self.load_pretrained_scope_weight(file_name, "resnet_v2_50", session)


class InceptionResnetV2(ModelBase):
    def __init__(self):
        ModelBase()
        return

    def build_model(self, image_batch, target, num_class, lambda_decay,
                    training):
        with slim.arg_scope(
                model.inception_resnet_v2.inception_resnet_v2_arg_scope(
                        batch_norm_decay=0.9)):
            _, end_points = model.inception_resnet_v2.inception_resnet_v2(
                image_batch, 1001, training)
            flatten = tf.layers.flatten(end_points['global_pool'])

        scope_name = "plant_seedings_build_inception_resnet_v2_with_lsoftmax"
        with tf.variable_scope(scope_name):
            linear = model.l_softmax.l_softmax(flatten, target, num_class, 4,
                                               lambda_decay, training,
                                               'l_softmax')

            logits = tf.nn.softmax(linear, name='softmax')
        return linear, logits, None

    def load_pretrained_weight(self, file_name, session):
        self.load_pretrained_scope_weight(file_name, "InceptionResnetV2", session)