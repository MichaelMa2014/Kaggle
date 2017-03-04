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

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib

if platform.system().find('Darwin') != -1:
    matplotlib.use('macosx')
else:
    matplotlib.use('agg')
import matplotlib.pyplot as plt

import util.preprocess as pre
import util.audit as audit
from util import INPUT_PATH, PROCESS_NUM, listdir_no_hidden


def main():
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()

    # audit.folder_name_equals_id()
    # audit.slice_size_is_512_by_512()
    audit.slope_and_intercept_consistent()

    # first_path = INPUT_PATH + '/' + patients[0]
    # first_slice = pre.load_pixel(first_path)[0]
    # mean = np.mean(first_slice)
    # std = np.std(first_slice)
    # min = np.min(first_slice)
    # max = np.max(first_slice)
    # first_slice[first_slice == min] = mean
    # first_slice[first_slice == max] = mean
    # first_slice = (first_slice - mean) / std
    # fig, ax = plt.subplots()
    # ax.imshow(first_slice, cmap='gray')
    # plt.hist(first_pixel.flatten(), bins=200)
    # plt.show()
    # first_pixel_resampled, new_spacing = pre.resample(first_pixel, first_slices, [5, 5, 5])
    # pre.plot_3d(first_pixel, 400)
    # pre.plot_3d(first_pixel_resampled, 400)
    # pre.plot_3d(first_pixel, 0)
    # pre.plot_3d(first_pixel_resampled, 0)

    # first_pixel = pre.load_pixel(first_path)
    # print(first_pixel.shape)

    # for i in range(PROCESS_NUM):
    #     p = multiprocessing.Process(target=pre.dcm_to_npy, args=(patients[i::PROCESS_NUM],))
    #     p.start()

    # labels = pd.read_csv(PATH + '/data/stage1_labels.csv')
    # for patient in patients:
    #     path = input_folder + patient
    #     pixel = pre.load_pixel(path)
    #
    #     print('%s scan shape' % patient, pixel.shape)
    #     try:
    #         label = labels[labels.id == patient].iloc[0, 1]
    #         print('%s is %s' % (patient, 'healthy' if label == 0 else 'cancerous'))
    #     except IndexError:
    #         print('%s is %s' % (patient, 'unknown'))


if __name__ == "__main__":
    main()
