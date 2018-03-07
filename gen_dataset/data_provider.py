# -*- coding: utf-8 -*-
"""
decode batch example from tfrecord

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import io
import numpy as np
import os
from PIL import Image
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
    num_batching_threads=8, queue_capacity=4000, min_after_dequeue=1000)

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.
    For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
        image: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.
    Returns:
        the centered image.
    Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
           than three or if the number of channels in `image` doesn't match the
           number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

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


def _read_and_decode(tfrecord_file_pattern, channel_num, image_size, preprocessing='inception'):
    """
    decode tfrecord samples from tfrecord file according file pattern
    decoded image will resize to image_size
    :param tfrecord_file_pattern:
    :param channel_num:
    :param image_size:
    :return: resized image tensor and label tensor
    """
    print('_read_and_decode preprocessing {0}'.format(preprocessing))
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

    if preprocessing == 'vgg':
        img = tf.to_float(img)
        img = tf.image.resize_images(img, image_size)  # float32 [0, 1]
        img = img * 255 #[0, 255]
        img = _mean_image_subtraction(img, [_R_MEAN, _G_MEAN, _B_MEAN])
    else:
        img = tf.image.resize_images(img, image_size)  # float32 [0, 1]
        img = tf.cast(img, tf.float32) * (1.0 / 255) * 2.0 - 1.0  #[-1, +1] float32

    return img, features['label']


def config_to_prefetch_queue(config=None,
                             dataset_dir=None,
                             batch_size=64,
                             random_flip_rot_train=False,
                             preprocessing='inception'):
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
        config['image_shape'][0:2], preprocessing)

    if random_flip_rot_train:
        image_train = tf.image.random_flip_up_down(image_train)

        k = tf.random_uniform([1], 1, 5, tf.int32)
        image_train = tf.image.rot90(image_train, k[0])

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
        config['image_shape'][0:2], preprocessing)

    image_test_batch, label_test_batch = tf.train.batch(
        [image_test, label_test],
        batch_size=batch_size,
        num_threads=shuffle_config.num_batching_threads,
        capacity=shuffle_config.queue_capacity)

    test_queue = slim.prefetch_queue.prefetch_queue(
        [image_test_batch, label_test_batch])

    return train_queue, test_queue


def tfrecord_file_to_nparray(tfrecord_file_name, image_size, preprocessing='inception'):
    print('tfrecord_file_to_nparray preprocessing {0}'.format(preprocessing))
    record_iterator = tf.python_io.tf_record_iterator(tfrecord_file_name)

    result = []

    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)

        encoded = example.features.feature[
            'image_plant/encoded'].bytes_list.value[0]
        image = Image.open(io.BytesIO(encoded))
        image = np.array(image.resize(image_size, Image.ANTIALIAS))[:, :, 0:3]

        if preprocessing == 'vgg':
            image[:, :, 0] = image[:, :, 0] - _R_MEAN
            image[:, :, 1] = image[:, :, 1] - _G_MEAN
            image[:, :, 2] = image[:, :, 2] - _B_MEAN
            image = image.astype(np.float32)
        else:
            image = image * (1.0 / 255) * 2.0 - 1.0  #[-1, +1] float32
            image = image.astype(np.float32)

        label = example.features.feature['label'].int64_list.value[0]

        result.append((image, label))

    return result
