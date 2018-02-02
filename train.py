#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import argparse
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

NUM_CLASS = 12

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="fine tune with mobilenet", default=96)
parser.add_argument("--num_epoch", help="fine tune with mobilenet", default=999)
parser.add_argument(
    "--early_stopping_step", help="fine tune with mobilenet", default=50)
args = parser.parse_args()


def train():
    train_queue, test_queue = data_provider.config_to_prefetch_queue(
        TRAIN_CONFIG,
        './gen_dataset',
        batch_size=args.batch_size,
        random_flip_rot_train=True)

    image_batch, label_batch = train_queue.dequeue()
    test_image_batch, test_label_batch = test_queue.dequeue()

    x = tf.placeholder(tf.float32, shape=(None, 256, 256, 3), name='x')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    is_training = tf.placeholder(tf.bool, name='phase')

    # [-1, +1] => [0, +1]
    image_to_summary = (image_batch + 1) / 2
    tf.summary.image('plant', image_to_summary, max_outputs=8)

    linear, logits, trainable_var = build_model.build_cnn_8_crelu_classifier_with_dropout(
        x, NUM_CLASS, is_training)

    loss_op = build_model.build_loss(y, linear)
    tf.summary.scalar("total_loss", loss_op)

    global_step = tf.train.create_global_step()
    train_op = build_model.build_train_op(loss_op, trainable_var, global_step)

    for var in tf.global_variables():
        tf.summary.histogram(var.op.name, var)

    correct_prediction = tf.equal(tf.squeeze(y), tf.argmax(linear, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('batch_accuracy', accuracy_op)
    confusion_matrix_op = tf.confusion_matrix(
        y, tf.argmax(linear, 1), num_classes=NUM_CLASS, dtype=tf.int32)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        session.run(tf.global_variables_initializer())

        # epoch
        training_process(accuracy_op, global_step, image_batch, label_batch,
                         is_training, loss_op, session, train_op, x, y,
                         args.num_epoch, confusion_matrix_op)

        print("thread.join")
        coord.request_stop()
        coord.join(threads)


def training_process(accuracy_op, global_step, image_batch, label_batch,
                     is_training, loss_op, session, train_op, x, y, num_epoch,
                     confusion_matrix_op):

    best_test_accuracy = 0

    merge_summary = tf.summary.merge_all()

    model_saver = tf.train.Saver(max_to_keep=15)
    save_path = 'model_weight'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    summary_writer = tf.summary.FileWriter(save_path, session.graph)

    best_model_save_path = 'model_best_weight'
    if not os.path.exists(best_model_save_path):
        os.makedirs(best_model_save_path)
    best_model_saver = tf.train.Saver(max_to_keep=10)

    test_data = data_provider.tfrecord_file_to_nparray(
        './gen_dataset/plant.config.test.tfrecord',
        TRAIN_CONFIG['image_shape'][0:2])

    early_stop_step = 0
    for i in range(num_epoch):
        training_phase(accuracy_op, global_step, i, image_batch, label_batch,
                       is_training, loss_op, merge_summary, session,
                       summary_writer, train_op, x, y)

        model_saver.save(
            session,
            os.path.join(save_path, "plant_seedings_classifier.ckpt"),
            global_step=global_step)

        best_acc_updated, best_test_accuracy = test_phase(
            accuracy_op, best_test_accuracy, is_training, session, test_data, x,
            y, confusion_matrix_op)

        if best_acc_updated:
            early_stop_step = 0
            print('*** save best model...')
            best_model_saver.save(
                session,
                os.path.join(best_model_save_path,
                             "plant_seedings_classifier_{0:.4f}.ckpt".format(
                                 best_test_accuracy)),
                global_step=global_step)
        else:
            early_stop_step += 1

        if early_stop_step >= args.early_stopping_step:
            print("early stop...")
            return


def test_phase(accuracy_op, best_test_accuracy, is_training, session, test_data,
               x, y, confusion_matrix_op):

    # for k in range(
    #        int(math.ceil(TRAIN_CONFIG['test_size'] / BATCH_SIZE))):
    #    images, labels = session.run(
    #        [test_image_batch, test_label_batch])
    #    accuracy_value = session.run(
    #        [accuracy],
    #        feed_dict={
    #            x: images,
    #            y: labels,
    #            training: False
    #        })
    #    test_accuracy_avg = test_accuracy_avg + (
    #        accuracy_value[0] - test_accuracy_avg) / (
    #            k + 1)
    # print("prefetch_queue acc", test_accuracy_avg)

    confusion_matrix = np.zeros((NUM_CLASS, NUM_CLASS))
    k = 0
    test_accuracy_avg = 0
    for image, label in test_data:
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        label = label.astype(np.int32)
        accuracy_value, confusion = session.run(
            [accuracy_op, confusion_matrix_op],
            feed_dict={
                x: image,
                y: label,
                is_training: False
            })
        test_accuracy_avg = test_accuracy_avg + (
            accuracy_value - test_accuracy_avg) / (
                k + 1)
        k += 1
        confusion_matrix = confusion_matrix + confusion
        sys.stdout.write("\rtest acc:{0}           ".format(test_accuracy_avg))
        sys.stdout.flush()
    print("")
    print(confusion_matrix)
    if best_test_accuracy < test_accuracy_avg:
        best_test_accuracy = test_accuracy_avg
        return True, best_test_accuracy
    return False, best_test_accuracy


def training_phase(accuracy_op, global_step, epoch, image_batch, label_batch,
                   is_training, loss_op, merge_summary, session, summary_writer,
                   train_op, x, y):
    accuracy_avg = 0.0
    for j in range(
            int(
                math.ceil(
                    TRAIN_CONFIG['training_size'] * 16 / args.batch_size))):
        images, labels = session.run([image_batch, label_batch])
        if j == 0:
            # step, summary, loss_value, accuracy_value, confusion, _ = session.run(
            #    [global_step, merge_summary, loss, accuracy, confusion_matrix_op,
            #     train_op])
            step, summary, loss_value, accuracy_value, _ = session.run(
                [global_step, merge_summary, loss_op, accuracy_op, train_op],
                feed_dict={
                    x: images,
                    y: labels,
                    is_training: True
                })
            summary_writer.add_summary(summary, step)
        else:
            # loss_value, accuracy_value, confusion, _ = session.run(
            #    [loss, accuracy, confusion_matrix_op, train_op])
            loss_value, accuracy_value, _ = session.run(
                [loss_op, accuracy_op, train_op],
                feed_dict={
                    x: images,
                    y: labels,
                    is_training: True
                })

        # confusion_matrix = confusion_matrix + confusion
        accuracy_avg = accuracy_avg + (accuracy_value - accuracy_avg) / (j + 1)
        sys.stdout.write("\r{0}--{1} training loss:{2}    ".format(
            epoch, j, loss_value))
        sys.stdout.flush()
    print("")
    print("training acc:{0}".format(accuracy_avg))
    # print(confusion_matrix)


train()
