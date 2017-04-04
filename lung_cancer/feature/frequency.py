# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from util import INPUT_PATH, OUTPUT_PATH
from util.preprocess import load_pixels_by_patient


def find_max_mask(pixel, n=10):
    ones = np.ones([n, n], dtype=np.int16)
    max_sum = 0
    max_i = 0
    max_j = 0
    for i in range(pixel.shape[0] - n):
        for j in range(pixel.shape[1] - n):
            mask = np.zeros(pixel.shape)
            mask[i:ones.shape[0] + i, j:ones.shape[1] + j] = ones
            temp = pixel.copy()
            temp[mask == 0] = 0
            if np.sum(temp) > max_sum:
                max_i = i
                max_j = j
                max_sum = np.sum(temp)
    return max_i, max_j, max_sum


def differentiate(pixel, threshold=700):
    upper = pixel[:-1, 1:-1].copy()
    lower = pixel[1:, 1:-1].copy()
    left = pixel[1:-1, :-1].copy()
    right = pixel[1:-1, 1:].copy()
    center_minus_up = (lower - upper)[:-1, :]
    center_minus_down = (upper - lower)[1:, :]
    center_minus_left = (right - left)[:, :-1]
    center_minus_right = (left - right)[:, 1:]
    sum_pixel = center_minus_up + center_minus_down + center_minus_left + center_minus_right
    sum_pixel = np.abs(sum_pixel)
    sum_pixel[sum_pixel < threshold] = 0
    sum_pixel[sum_pixel >= threshold] = 1
    return sum_pixel


def multi_detect_with_mask(patients):
    for patient in patients:
        out_file = open(OUTPUT_PATH + '/' + patient + '_mask.csv', 'w')
        out_file.write('patient_id,pixel_id,max_mask,i,j\n'.encode('utf-8'))
        out_file.flush()
        pixels = load_pixels_by_patient(patient)
        for i in range(0, len(pixels)):
            print(i)
            max_i, max_j, max_sum = detect_with_mask(pixels[i])
            out_file.write('%s,%s,%s,%s,%s\n'.encode('utf-8') % (patient, i, max_sum, max_i, max_j))
            out_file.flush()
        out_file.close()


def detect_with_mask(pixel):
    n = 10
    sum_pixel = differentiate(pixel)
    max_i, max_j, max_sum = find_max_mask(sum_pixel, n)
    return max_i, max_j, max_sum  # should be n * n if all pixels under the mask is turned on


def load_diffs_by_patient(patient):
    """
    Load the diffs of pixels (np array) saved for this patient, if not found, try to construct the np array and save it
    :param patient: patient id
    :return:
    """
    path = INPUT_PATH + '/' + patient
    if not os.path.exists(path + '/' + patient + '_diff.npy'):
        pixels = load_pixels_by_patient(patient)
        diffs = []
        for pixel in pixels:
            diffs.append(differentiate(pixel))
        diffs = np.array(diffs)
        np.save(path + '/' + patient + '_diff', diffs)
    else:
        diffs = np.load(path + '/' + patient + '_diff.npy')
    return diffs


def multi_load_diffs_by_patient(patients):
    for patient in patients:
        diffs = load_diffs_by_patient(patient)
        print(diffs.shape)


def plot_mask(pixel, patient="Unknown", i=0):
    """
    Draw 4 subplots, explore 2-Means
    :param pixel: np array
    :param patient: patient id for output
    :param i: pixel id for output
    """
    fig, axs = plt.subplots(2, 2)
    for ax_row in axs:
        for _ax in ax_row:
            _ax.axis('off')

    ax = axs[0]

    ax[0].set_title('Original')
    cax = ax[0].imshow(pixel)
    fig.colorbar(cax, ax=ax[0])

    my_min = np.min(pixel)
    pixel[pixel == my_min] = -1000  # This is to cut off the margin left by the machine
    middle = np.reshape(pixel[100:400, 100:400], [-1, 1])
    k_means = KMeans(n_clusters=2).fit(middle)
    threshold = np.mean(k_means.cluster_centers_)
    ax[1].set_title('Apply 2-Means after pulling up')
    cax = ax[1].imshow(np.where(pixel < threshold, [0], [1]))
    fig.colorbar(cax, ax=ax[1])

    ax = axs[1]

    ax[0].set_title('Diff')
    diff = differentiate(pixel, 700)
    cax = ax[0].imshow(diff, cmap="Paired")
    fig.colorbar(cax, ax=ax[0])

    ax[1].set_title('Threshold diff')
    diff[(pixel < threshold)[1:-1, 1:-1]] = 0
    cax = ax[1].imshow(diff, cmap="Paired")
    fig.colorbar(cax, ax=ax[1])

    fig.savefig(OUTPUT_PATH + '/%s_%s' % (patient, i), dpi=400)
