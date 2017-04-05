# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation


def fft(input_shape):
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer="sgd", loss="categorical_crossentropy")

    return model

