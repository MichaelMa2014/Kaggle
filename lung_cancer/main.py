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
from sklearn.cluster import KMeans
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import util.preprocess as pre
import util.audit as audit
from util import INPUT_PATH, OUTPUT_PATH, PROCESS_NUM, listdir_no_hidden


def main():
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()

    # audit.folder_name_equals_id()
    # audit.slice_size_is_512_by_512()
    # audit.slope_and_intercept_consistent()

    first_patient = patients[1]
    first_path = INPUT_PATH + '/' + first_patient
    slices = pre.load_pixel(first_path)
    for i in range(0, len(slices)):
        first_slice = slices[i]
        fig, ax = plt.subplots(2, 2)

        middle = np.reshape(first_slice[100:400, 100:400], [-1, 1])
        kmeans = KMeans(n_clusters=2).fit(middle)
        threshold = np.mean(kmeans.cluster_centers_)
        cax = ax[0, 0].imshow(first_slice)
        fig.colorbar(cax, ax=ax[0, 0])
        ax[0, 1].imshow(np.where(first_slice < threshold, 0, 1))

        mean = np.mean(first_slice)
        std = np.std(first_slice)
        min = np.min(first_slice)
        max = np.max(first_slice)
        first_slice[first_slice == min] = mean
        first_slice[first_slice == max] = mean
        first_slice = (first_slice - mean) / std

        middle = np.reshape(first_slice[100:400, 100:400], [-1, 1])
        kmeans = KMeans(n_clusters=2).fit(middle)
        threshold = np.mean(kmeans.cluster_centers_)
        cax = ax[1, 0].imshow(first_slice)
        fig.colorbar(cax, ax=ax[1, 0])
        ax[1, 1].imshow(np.where(first_slice < threshold, 0, 1))

        plt.savefig(OUTPUT_PATH + '/%s_%s' % (first_patient, i))
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
