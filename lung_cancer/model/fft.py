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
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation

import util.preprocess as pre
from util import INPUT_PATH, OUTPUT_PATH, PROCESS_NUM, listdir_no_hidden, labels
from util.log import _INFO, _ERROR


def classifier(input_shape):
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer="sgd", loss="categorical_crossentropy")

    return model


def train():
    """
    Train and save FFT -> LR
    """
    _INFO("fft training started")
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()
    _INFO("Found %s patients" % len(patients))

    model = classifier((512, 257))
    keras.utils.plot_model(model, to_file=OUTPUT_PATH + "/fft.png", show_shapes=True)

    pos_count = 0
    neg_count = 0

    for patient in patients:
        _INFO("fft dealing with patient " + patient)
        segments = pre.load_segment_by_patient(patient)

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

        segments = np.fft.rfft2(segments)
        segments = np.abs(segments)

        _INFO("x_train shape " + str(segments.shape))
        _INFO("y_train shape " + str(label.shape))
        model.fit(segments, label, epochs=1, verbose=2)
    model.save(OUTPUT_PATH + "/fft_balance.h5")

