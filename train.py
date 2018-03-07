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
from tensorflow.python import debug as tf_debug
import horovod.tensorflow as hvd

import model.build_model as build_model
import gen_dataset.data_provider as data_provider

hvd.init()

TRAIN_CONFIG = {
    'name': 'plant_seedings_classification',
    'training_size': 3800,
    'test_size': 950,
    'pattern_training_set': 'plant.config.train*.tfrecord',
    'pattern_test_set': 'plant.config.test*.tfrecord',
    'image_shape': (256, 256, 3),
    'items_to_descriptions': {''}
}

RESNET_50_TRAIN_CONFIG = {
    'name': 'plant_seedings_classification',
    'training_size': 3800,
    'test_size': 950,
    'pattern_training_set': 'plant.config.train*.tfrecord',
    'pattern_test_set': 'plant.config.test*.tfrecord',
    'image_shape': (224, 224, 3),
    'items_to_descriptions': {''}
}

NUM_CLASS = 12
CONFIG_USE = RESNET_50_TRAIN_CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="", default=196, type=int)
parser.add_argument("--num_epoch", help="", default=999, type=int)
parser.add_argument(
    "--early_stopping_step", help="", default=100, type=int)
parser.add_argument(
    "--lambda_decay_init", help="", default=1000.0, type=float)
parser.add_argument(
    "--lambda_decay_steps", help="", default=2000, type=int)
parser.add_argument(
    "--lambda_decay_rate", help="", default=0.8, type=float)
parser.add_argument(
    "--lambda_decay_min", help="", default=9.0, type=float)
parser.add_argument(
    "--tfdbg", help="", default=False, type=bool)
args = parser.parse_args()

BATCH_SIZE_PER_REPLICA = int(args.batch_size / hvd.size())
DATA_AUGMENTATION_TIMES_PER_REPLICA = int(8 / hvd.size())

def test_augmented_acc(linear, y):
    linear = build_model.build_test_time_vote(linear)
    test_correct_prediction = tf.equal(tf.squeeze(y), tf.argmax(linear, 1))
    test_accuracy_op = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
    confusion_matrix_op = tf.confusion_matrix(
        y, tf.argmax(linear, 1), num_classes=NUM_CLASS, dtype=tf.int32)
    return test_accuracy_op, confusion_matrix_op

def train():
    train_queue, test_queue = data_provider.config_to_prefetch_queue(
        CONFIG_USE,
        './gen_dataset',
        batch_size=BATCH_SIZE_PER_REPLICA,
        random_flip_rot_train=True,
        preprocessing='inception')

    image_batch, label_batch = train_queue.dequeue()
    test_image_batch, test_label_batch = test_queue.dequeue()

    x = tf.placeholder(tf.float32, shape=(None, CONFIG_USE['image_shape'][0],
        CONFIG_USE['image_shape'][1], CONFIG_USE['image_shape'][2]), name='x')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    is_training = tf.placeholder(tf.bool, name='phase')

    # [-1, +1] => [0, +1]
    image_to_summary = (image_batch + 1) / 2
    tf.summary.image('plant', image_to_summary, max_outputs=8)

    global_step = tf.train.create_global_step()

    lambda_decay = tf.train.exponential_decay(args.lambda_decay_init, global_step,
                                               args.lambda_decay_steps,
                                               args.lambda_decay_rate,
                                               staircase=True, name='lambda_decay')

    lambda_decay = tf.cond(lambda_decay > args.lambda_decay_min,
                           lambda: lambda_decay,
                           lambda: tf.constant(args.lambda_decay_min))

    tf.summary.scalar("lambda_decay", lambda_decay)

    linear, logits, trainable_var = build_model.build_resnet_v2_50(
        x, y, NUM_CLASS, lambda_decay, is_training)

    loss_op = build_model.build_loss(y, linear)
    tf.summary.scalar("total_loss", loss_op)

    train_op = build_model.build_train_op(loss_op, trainable_var, global_step)

    for var in tf.global_variables():
        tf.summary.histogram(var.op.name, var)

    correct_prediction = tf.equal(tf.squeeze(y), tf.argmax(linear, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('batch_accuracy', accuracy_op)
    test_augmented_accuracy_op, test_augmented_confusion_matrix_op = test_augmented_acc(linear, y)
    confusion_matrix_op = tf.confusion_matrix(
        y, tf.argmax(linear, 1), num_classes=NUM_CLASS, dtype=tf.int32)
    x_test = tf.placeholder(tf.float32, shape=(None, CONFIG_USE['image_shape'][0],
        CONFIG_USE['image_shape'][1], CONFIG_USE['image_shape'][2]), name='x_test')
    x_augmented = build_model.build_test_time_data_augmentation(x_test)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    with tf.Session(config=session_config) as session:
        if args.tfdbg:
            session = tf_debug.LocalCLIDebugWrapperSession(session)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        session.run(tf.global_variables_initializer())
        session.run(hvd.broadcast_global_variables(0))

        # epoch
        training_process(accuracy_op, global_step, image_batch, label_batch,
                         is_training, loss_op, session, train_op, x, y,
                         args.num_epoch, confusion_matrix_op,
                         test_augmented_accuracy_op,
                         test_augmented_confusion_matrix_op, x_test, x_augmented)

        print("thread.join")
        coord.request_stop()
        coord.join(threads)


def training_process(accuracy_op, global_step, image_batch, label_batch,
                     is_training, loss_op, session, train_op, x, y, num_epoch,
                     confusion_matrix_op, test_augmented_accuracy_op,
                     test_augmented_confusion_matrix_op, x_test, x_augmented):

    best_test_accuracy = 0

    merge_summary = tf.summary.merge_all()

    model_saver = tf.train.Saver(max_to_keep=15)
    save_path = 'model_weight_{0}'.format(hvd.rank())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    summary_writer = tf.summary.FileWriter(save_path, session.graph)

    best_model_save_path = 'model_best_weight'
    if not os.path.exists(best_model_save_path):
        os.makedirs(best_model_save_path)
    best_model_saver = tf.train.Saver(max_to_keep=10)

    test_data = data_provider.tfrecord_file_to_nparray(
        './gen_dataset/plant.config.test.tfrecord',
        CONFIG_USE['image_shape'][0:2],
        preprocessing='inception')

    build_model.restore_pretrained_resnet_v2_50(session)

    early_stop_step = 0
    for i in range(num_epoch):
        loss_avg = training_phase(accuracy_op, global_step, i, image_batch, label_batch,
                       is_training, loss_op, merge_summary, session,
                       summary_writer, train_op, x, y)

        model_saver.save(
            session,
            os.path.join(save_path, "plant_seedings_classifier_{0:.8f}.ckpt".format(loss_avg)),
            global_step=global_step)

        if hvd.rank() == 0:
            best_acc_updated, best_test_accuracy = test_phase(
                accuracy_op, best_test_accuracy, is_training, session, test_data, x,
                y, confusion_matrix_op, test_augmented_accuracy_op,
                test_augmented_confusion_matrix_op, x_test, x_augmented)

            if best_acc_updated:
                early_stop_step = 0
                print('========================= save best model =======================')
                best_model_saver.save(
                    session,
                    os.path.join(best_model_save_path,
                                "plant_seedings_classifier_{0:.4f}_{1:.8f}.ckpt".format(
                                    best_test_accuracy, loss_avg)),
                    global_step=global_step)
            else:
                early_stop_step += 1

            if early_stop_step >= args.early_stopping_step:
                print("early stop...")
                return


def test_phase(accuracy_op, best_test_accuracy, is_training, session, test_data,
               x, y, confusion_matrix_op, test_augmented_accuracy_op,
               test_augmented_confusion_matrix_op, x_test, x_augmented):

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
    augmented_confusion_matrix = np.zeros((NUM_CLASS, NUM_CLASS))
    k = 0
    test_accuracy_avg = 0
    test_augmented_accuracy_avg = 0
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

        image_augmented = session.run([x_augmented], feed_dict={x_test: image})[0]
        augmented_accuracy_value, augmented_confusion = session.run(
            [test_augmented_accuracy_op, test_augmented_confusion_matrix_op],
            feed_dict={
                x: image_augmented,
                y: label,
                is_training: False
            })

        test_augmented_accuracy_avg = test_augmented_accuracy_avg + (
            augmented_accuracy_value - test_augmented_accuracy_avg) / (
                k + 1)

        k += 1
        confusion_matrix = confusion_matrix + confusion
        augmented_confusion_matrix = augmented_confusion_matrix + augmented_confusion
        sys.stdout.write("\raugmented_test acc:{0} - test acc:{1}           ".format(test_augmented_accuracy_avg, test_accuracy_avg))
        sys.stdout.flush()
    print("")
    print("confusion_matrix")
    print(confusion_matrix)
    print("augmented_confusion_matrix")
    print(augmented_confusion_matrix)

    if best_test_accuracy < test_accuracy_avg:
        print("{0} < {1}".format(best_test_accuracy, test_accuracy_avg))
        best_test_accuracy = test_accuracy_avg
        return True, best_test_accuracy
    print("{0} > {1}".format(best_test_accuracy, test_accuracy_avg))
    return False, best_test_accuracy


def training_phase(accuracy_op, global_step, epoch, image_batch, label_batch,
                   is_training, loss_op, merge_summary, session, summary_writer,
                   train_op, x, y):
    accuracy_avg = 0.0
    loss_avg = 0.0
    for j in range(
            int(
                math.ceil(
                    CONFIG_USE['training_size'] * DATA_AUGMENTATION_TIMES_PER_REPLICA / BATCH_SIZE_PER_REPLICA))):
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
        loss_avg = loss_avg + (loss_value - loss_avg) / (j + 1)
        sys.stdout.write("\r{0}--{1} training avg loss:{2} batch loss:{3}    ".format(
            epoch, j, loss_avg, loss_value))
        sys.stdout.flush()
    print("")
    print("training acc:{0}".format(accuracy_avg))
    return loss_avg


train()
