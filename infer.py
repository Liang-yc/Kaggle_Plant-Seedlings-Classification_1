#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import numpy as np
from PIL import Image
import tensorflow as tf

from model.model_class import InceptionResnetV2, Resnet50V2

IndexToClass = {
    0: 'Black-grass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickweed',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small-flowered Cranesbill',
    11: 'Sugar beet',
}

def preprocess_files(size):
    files = glob.glob("raw_dataset/test/*.png")
    result = []
    for file_path in files:
        image = Image.open(file_path)
        image = image.resize((size, size), Image.ANTIALIAS)

        # [-1, +1]
        array = ((np.asarray(image, dtype=np.float32) / 255) - 0.5) * 2

        assert(array.shape[-1] == 3)
        assert (array.shape[0] == size)

        file_name = os.path.split(file_path)[-1]
        result.append((file_name, array))

    return result

def infer():
    model = InceptionResnetV2()

    x = tf.placeholder(
        tf.float32,
        shape=(None, model.get_input_shape()[0], model.get_input_shape()[1],
               model.get_input_shape()[2]),
        name='x')
    is_training = tf.placeholder(tf.bool, name='phase')

    # place holder, not use in infer
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    num_class = 12
    lambda_decay = tf.constant(9.0)

    linear, logits, _ = model.build_model(x, y, num_class, lambda_decay,
                                          training=is_training)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    result = []
    with tf.Session(config=session_config) as session:
        session.run(tf.global_variables_initializer())
        model.restore_weight('performance_log_01/inception_resnet/plant_seedings_classifier_0.9811_0.12606539.ckpt-80518', session)

        file_name_images = preprocess_files(model.get_input_shape()[0])
        for file_name, image in file_name_images:
            image = np.expand_dims(image, axis=0)
            _, probability = session.run([linear, logits],
                                         feed_dict={x: image, y: 0,
                                                    is_training: False})
            class_name = IndexToClass[np.argmax(probability)]

            line = '{0},{1}\n'.format(file_name, class_name)
            result.append(line)
            print(line)

    with open('submission.csv', 'w') as submission:
        result.sort()
        submission.writelines(result)

infer()