#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

import model.build_model as build_model
import gen_dataset.data_provider as data_provider

TRAIN_CONFIG = {
    'name': 'plant_seedings_classification',
    'training_size': 3800,
    'test_size': 950,
    'pattern_training_set': 'plant.config.train*.tfrecord',
    'pattern_test_set': 'plant.config.test*.tfrecord',
    'image_shape': (256, 256, 3),
    'items_to_descriptions': {''}
}

BATCH_SIZE = 64
NUM_CLASS = 12
NUM_EPOCH = 20

def train():
    train_queue, test_queue = data_provider.config_to_prefetch_queue(
        TRAIN_CONFIG, './gen_dataset', batch_size=BATCH_SIZE)

    image_batch, label_batch = train_queue.dequeue()

    # [-1, +1] => [0, +1]
    image_to_summary = (image_batch + 1) / 2
    tf.summary.image('plant', image_to_summary, max_outputs=8)

    linear, logits, trainable_var = build_model.build_classifier(
        image_batch, NUM_CLASS)

    loss = build_model.build_loss(label_batch, linear)
    tf.summary.scalar("total_loss", loss)

    global_step = tf.train.create_global_step()
    train_op = build_model.build_train_op(loss, trainable_var, global_step)

    for var in tf.global_variables():
        tf.summary.histogram(var.op.name, var)

    correct_prediction = tf.equal(tf.squeeze(label_batch), tf.argmax(linear, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('batch_accuracy', accuracy)
    confusion_matrix_op = tf.confusion_matrix(tf.squeeze(label_batch),
                                              tf.argmax(linear, 1))

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    model_saver = tf.train.Saver(max_to_keep=10)
    merge_summary = tf.summary.merge_all()

    save_path = 'model_weight'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(save_path, session.graph)

        # epoch
        for i in range(NUM_EPOCH):
            confusion_matrix = np.zeros((NUM_CLASS, NUM_CLASS))
            accuracy_avg = 0.0

            for j in range(10):
                if j == 0:
                    step, summary, loss_value, accuracy_value, confusion, _ = session.run(
                        [global_step, merge_summary, loss, accuracy, confusion_matrix_op,
                         train_op])
                    summary_writer.add_summary(summary, step)
                else:
                    loss_value, accuracy_value, confusion, _ = session.run(
                        [loss, accuracy, confusion_matrix_op, train_op])

                confusion_matrix = confusion_matrix + confusion
                accuracy_avg = accuracy_avg + (
                                              accuracy_value - accuracy_avg) / (
                                              j + 1)
                sys.stdout.write(
                    "\r{0}--{1} training accuracy(ma):{2}    ".format(i, j,
                                                                      accuracy_avg))
                sys.stdout.flush()

            print("")
            print(confusion_matrix)

            model_saver.save(session, os.path.join(save_path, "plant_seedings_classifier.ckpt"),
                       global_step=global_step)

        print("thread.join")
        coord.request_stop()
        coord.join(threads)



train()