# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import multiprocessing
import platform

import pandas as pd
import numpy as np
import matplotlib
import keras

if platform.system().find('Darwin') != -1:
    matplotlib.use('tkagg')
else:
    matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.rcParams['image.cmap'] = 'viridis'

import util.preprocess as pre
import util.audit as audit
from util import INPUT_PATH, OUTPUT_PATH, PROCESS_NUM, listdir_no_hidden, labels
from util.log import _INFO, _ERROR

import feature.frequency as fre
import model.cnn as cnn
import model.fft as fft


def main():
    _INFO("main started")
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()
    _INFO("Found %s patients" % len(patients))

    # for i in range(PROCESS_NUM):
    #     p = multiprocessing.Process(target=fre.multi_load_diffs_by_patient, args=(patients[i::PROCESS_NUM],))
    #     p.start()

    # model = cnn.classifier((512, 512, 1))
    model = fft.fft((512, 257))
    keras.utils.plot_model(model, to_file=OUTPUT_PATH + "/model.png", show_shapes=True)

    for patient in patients:
        segments = pre.load_segment_by_patient(patient)
        # segments = np.expand_dims(segments, axis=3)
        segments = np.fft.rfft2(segments)
        segments = np.abs(segments)
        _INFO(segments.shape)
        label = labels[patient]
        if label == 1:
            label = np.array([0, 1])
        else:
            label = np.array([1, 0])
        label = np.tile(label, (segments.shape[0], 1))
        model.fit(segments, label, epochs=1, verbose=2)
    # model.save(OUTPUT_PATH + "/cnn.h5")
    model.save(OUTPUT_PATH + "/fft.h5")

    # for i in range(PROCESS_NUM):
    #     p = multiprocessing.Process(target=fre.multi_detect_with_mask, args=(patients[i::PROCESS_NUM],))
    #     p.start()
    #
    # for patient in patients[0:1]:
    #     pixels = pre.load_pixels_by_patient(patient)
    #     for i in range(126, len(pixels)):
    #         fre.plot_mask(pixels[i], patient, i)
    #         break


if __name__ == "__main__":
    main()
