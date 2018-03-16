#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import numpy as np

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

ClassToIndex = {
    'Black-grass': 0,
    'Charlock': 1,
    'Cleavers': 2,
    'Common Chickweed': 3,
    'Common wheat': 4,
    'Fat Hen': 5,
    'Loose Silky-bent': 6,
    'Maize': 7,
    'Scentless Mayweed': 8,
    'Shepherds Purse': 9,
    'Small-flowered Cranesbill': 10,
    'Sugar beet': 11,
}

ensemble_result = dict()
files = glob.glob("ensemble/*.csv")
for file_path in files:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line == 'file,species':
                continue
            picture_name, category = line.split(',')
            assert(picture_name.endswith('png'))

            try:
                value = ensemble_result[picture_name]
            except KeyError:
                value = ensemble_result[picture_name] = np.zeros([12], dtype=np.int32)
            value[ClassToIndex[category]] += 1

with open('ensemble.csv', 'w') as ensemble_file:
    ensemble_file.write('file,species\n')
    result = []
    console_output = []
    for k, v in ensemble_result.items():
        console_output.append('{0},{1},{2}'.format(k,
                                   IndexToClass[np.argmax(v)], v[np.argmax(v)]))
        result.append('{0},{1}\n'.format(k,
                                   IndexToClass[np.argmax(v)]))
    console_output.sort()
    for line in console_output:
        print(line)
    result.sort()
    ensemble_file.writelines(result)




