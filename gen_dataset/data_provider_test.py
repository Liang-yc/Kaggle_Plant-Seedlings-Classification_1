# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image

import data_provider

TEST_CONFIG = {
    'name': 'test',
    'training_size': 1000,
    'test_size': 1000,
    'pattern_training_set': 'raw_data_train.config.tfrecord',
    'pattern_test_set': 'raw_data_train.config.tfrecord',
    'image_shape': (256, 256, 3),
    'items_to_descriptions': {''}
}


def test():
    train_queue, _ = data_provider.config_to_prefetch_queue(
        config=TEST_CONFIG, dataset_dir='./')
    face_batch, label_batch = train_queue.dequeue()

    init = tf.global_variables_initializer()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        session.run(init)
        for i in range(1):

            images, labels = session.run([face_batch, label_batch])

            for j, image in enumerate(images):
                # image [-1, +1] float32
                # => uint8
                image = (((image + 1) / 2) * 255).astype(np.uint8)
                im = Image.fromarray(image)
                im.save("{0}_{1}_plant.jpg".format(i, labels[j]))

        print("thread.join")
        coord.request_stop()
        coord.join(threads)


test()
