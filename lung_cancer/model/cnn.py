# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

import util.preprocess as pre
from util import INPUT_PATH, OUTPUT_PATH, PROCESS_NUM, listdir_no_hidden, labels
from util.log import _INFO, _ERROR


def classifier(input_shape, kernel_size=3, pool_size=2):
    model = Sequential()

    # 16 filters
    model.add(Conv2D(16, kernel_size, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(32, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(64, kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])
    return model


def train():
    """
    Train and save 9 steps -> CNN
    """
    _INFO("cnn training started")
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()
    _INFO("Found %s patients" % len(patients))

    model = classifier((512, 512, 1))
    keras.utils.plot_model(model, to_file=OUTPUT_PATH + "/cnn.png", show_shapes=True)

    pos_count = 0
    neg_count = 0

    for patient in patients:
        _INFO("cnn dealing with patient " + patient)
        segments = pre.load_segment_by_patient(patient)
        segments = np.expand_dims(segments, axis=3)

        label = labels[patient]
        if label == 1:
            label = np.array([0, 1])
            pos_count += segments.shape[0]
        else:
            label = np.array([1, 0])
            neg_count += segments.shape[0]
        if neg_count + 500 < pos_count:
            _INFO("Too many negative pixels, skipping patient " + patient)
            continue

        label = np.tile(label, (segments.shape[0], 1))

        _INFO("x_train shape " + str(segments.shape))
        _INFO("y_train shape " + str(label.shape))
        model.fit(segments, label, epochs=1, verbose=2)
    model.save(OUTPUT_PATH + "/cnn_balance.h5")

