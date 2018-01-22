# -*- coding: utf-8 -*-
"""
decode batch example from tfrecord

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.platform import gfile

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset')
DEFAULT_CONFIG = {
    'name': 'default',
    'training_size': 90000,
    'test_size': 900,
    'pattern_training_set': 'training*.tfrecord',
    'pattern_test_set': 'test*.tfrecord',
    'image_shape': (256, 256, 3),
    'items_to_descriptions': {''}
}
ShuffleBatchConfig = collections.namedtuple(
    'ShuffleBatchConfig',
    ['num_batching_threads', 'queue_capacity', 'min_after_dequeue'])
DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(
    num_batching_threads=8, queue_capacity=3000, min_after_dequeue=1000)


def config_to_slim_dataset(config=None, dataset_dir=None):
    """

    :param config:
    :param dataset_dir:
    :return:
    """

    if not dataset_dir:
        dataset_dir = DEFAULT_DATASET_DIR

    if not config:
        config = DEFAULT_CONFIG

    zero = tf.zeros([1], dtype=tf.int64)
    keys_to_features = {
        'image_plant/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image_plant/format':
            tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'label':
            tf.FixedLenFeature([1], tf.int64, default_value=zero),
    }

    items_to_handlers = {
        'image_plant':
            slim.tfexample_decoder.Image(
                shape=config['image_shape'],
                image_key='image_plant/encoded',
                format_key='image_plant/format'),
        'label':
            slim.tfexample_decoder.Tensor('label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)

    file_pattern_training_set = os.path.join(dataset_dir,
                                             config['pattern_training_set'])

    file_pattern_test_set = os.path.join(dataset_dir,
                                         config['pattern_test_set'])

    training_set = slim.dataset.Dataset(
        data_sources=file_pattern_training_set,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=config['training_size'],
        items_to_descriptions=config['items_to_descriptions'])

    test_set = slim.dataset.Dataset(
        data_sources=file_pattern_test_set,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=config['test_size'],
        items_to_descriptions=config['items_to_descriptions'])

    return training_set, test_set


def slim_dataset_to_prefetch_queue(dataset, batch_size, shuffle=True):
    """

    :param dataset:
    :param batch_size:
    :param shuffle:
    :return:
    """

    shuffle_config = DEFAULT_SHUFFLE_CONFIG

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=True,
        common_queue_capacity=16 * batch_size,
        common_queue_min=2 * batch_size)

    image_plant, label = provider.get(['image_plant', 'label'])

    if shuffle:
        image_plant_batch, label_batch = tf.train.shuffle_batch(
            [image_plant, label],
            batch_size=batch_size,
            num_threads=shuffle_config.num_batching_threads,
            capacity=shuffle_config.queue_capacity,
            min_after_dequeue=shuffle_config.min_after_dequeue)
    else:
        image_plant_batch, label_batch = tf.train.batch(
            [image_plant, label],
            batch_size=batch_size,
            num_threads=shuffle_config.num_batching_threads,
            capacity=shuffle_config.queue_capacity)

    # resize to 224 x 224 (h x w)
    #image_plant_batch = tf.cast(tf.image.resize_images(
    #    image_plant_batch, [224, 224]), tf.uint8)

    return slim.prefetch_queue.prefetch_queue([image_plant_batch, label_batch])


def _read_and_decode(tfrecord_file_pattern, channel_num, image_size):
    """
    decode tfrecord samples from tfrecord file according file pattern
    decoded image will resize to image_size
    :param tfrecord_file_pattern:
    :param channel_num:
    :param image_size:
    :return: resized image tensor and label tensor
    """
    filename_queue = tf.train.string_input_producer(
        gfile.Glob(tfrecord_file_pattern))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(
        filename_queue)  # return the file and the name of file
    features = tf.parse_single_example(
        serialized_example,  # see parse_single_sequence_example for sequence example
        features={
            'image_plant/format': tf.FixedLenFeature([], tf.string),
            'image_plant/encoded': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    # You can do more image distortion here for training data
    img = tf.image.decode_png(
        features['image_plant/encoded'], channels=channel_num,
        name='tf_decode')  # uint8
    img = tf.image.resize_images(img, image_size)  # float32
    img = tf.cast(img, tf.float32) * (1.0 / 255) * 2.0 - 1.0  #[-1, +1] float32

    return img, features['label']


def config_to_prefetch_queue(config=None, dataset_dir=None, batch_size=64):
    """
    read var size image saved in tfrecord, and resize it to config.image_shape
    :param config:
    :param dataset_dir:
    :return: train_queue with shuffle and test_queue without shuffle
    """

    if not dataset_dir:
        dataset_dir = DEFAULT_DATASET_DIR

    if not config:
        config = DEFAULT_CONFIG

    shuffle_config = DEFAULT_SHUFFLE_CONFIG

    image_train, label_train = _read_and_decode(
        os.path.join(dataset_dir, config['pattern_training_set']), 3,
        config['image_shape'][0:2])

    image_train_batch, label_train_batch = tf.train.shuffle_batch(
        [image_train, label_train],
        batch_size=batch_size,
        num_threads=shuffle_config.num_batching_threads,
        capacity=shuffle_config.queue_capacity,
        min_after_dequeue=shuffle_config.min_after_dequeue)

    train_queue = slim.prefetch_queue.prefetch_queue(
        [image_train_batch, label_train_batch])

    image_test, label_test = _read_and_decode(
        os.path.join(dataset_dir, config['pattern_test_set']), 3,
        config['image_shape'][0:2])

    image_test_batch, label_test_batch = tf.train.batch(
        [image_test, label_test],
        batch_size=batch_size,
        num_threads=shuffle_config.num_batching_threads,
        capacity=shuffle_config.queue_capacity)

    test_queue = slim.prefetch_queue.prefetch_queue(
        [image_test_batch, label_test_batch])

    return train_queue, test_queue
