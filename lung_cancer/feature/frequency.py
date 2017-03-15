# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from util import OUTPUT_PATH


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


def detect_with_mask(pixel):
    n = 10
    sum_pixel = differentiate(pixel)
    max_i, max_j, max_sum = find_max_mask(sum_pixel, n)
    return max_sum > 2 * n  # should be n * n if all pixels under the mask is turned on


def plot_mask(pixel, patient="Unknown", i=0):
    """
    Draw 4 subplots, explore 2-Means
    :param pixel: np array
    :param patient: patient id for output
    :param i: pixel id for output
    """
    fig, ax = plt.subplots(3, 2)

    ax[0, 0].axis('off')
    ax[0, 0].set_title('Original')
    cax = ax[0, 0].imshow(pixel)
    fig.colorbar(cax, ax=ax[0, 0])

    middle = np.reshape(pixel[100:400, 100:400], [-1, 1])
    k_means = KMeans(n_clusters=2).fit(middle)
    threshold = np.mean(k_means.cluster_centers_)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('Apply 2-Means to original')
    ax[0, 1].imshow(np.where(pixel < threshold, 0, 1))

    my_mean = np.mean(pixel)
    my_min = np.min(pixel)
    pixel[pixel == my_min] = -1000  # This is to cut off the margin left by the machine
    ax[1, 0].axis('off')
    ax[1, 0].set_title('Pull up minimum to -1000 (air)')
    cax = ax[1, 0].imshow(pixel)
    fig.colorbar(cax, ax=ax[1, 0])

    middle = np.reshape(pixel[100:400, 100:400], [-1, 1])
    k_means = KMeans(n_clusters=2).fit(middle)
    threshold = np.mean(k_means.cluster_centers_)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Apply 2-Means after pulling up')
    ax[1, 1].imshow(np.where(pixel < threshold, 0, 1))

    # std = np.std(pixel)
    # pixel = (pixel.astype(np.float64) - my_mean) / std

    upper = pixel[:-1, 1:-1].copy()
    lower = pixel[1:, 1:-1].copy()
    left = pixel[1:-1, :-1].copy()
    right = pixel[1:-1, 1:].copy()
    center_minus_up = (lower - upper)[:-1, :]
    center_minus_down = (upper - lower)[1:, :]
    center_minus_left = (right - left)[:, :-1]
    center_minus_right = (left - right)[:, 1:]
    sum_pixel = center_minus_up + center_minus_down + center_minus_left + center_minus_right

    ax[2, 0].axis('off')
    ax[2, 0].set_title('Central minus peripheral')
    cax = ax[2, 0].imshow(sum_pixel)
    fig.colorbar(cax, ax=ax[2, 0])

    ax[2, 1].axis('off')
    ax[2, 1].set_title('Threshold diff')
    sum_pixel = differentiate(pixel, 700)
    sum_pixel[(pixel < threshold)[1:-1, 1:-1]] = 0

    cax = ax[2, 1].imshow(sum_pixel, cmap=plt.cm.Paired)
    fig.colorbar(cax, ax=ax[2, 1])

    fig.savefig(OUTPUT_PATH + '/%s_%s' % (patient, i), dpi=400)

    fig2, ax2 = plt.subplots(1, 2)

    ax2[1].axis('off')
    ax2[1].set_title('Threshold diff')
    cax = ax2[0].imshow(sum_pixel, cmap=plt.cm.Paired)
    fig2.colorbar(cax, ax=ax2[0])

    n = 10
    max_i, max_j, max_sum = find_max_mask(sum_pixel, n)

    ones = np.ones([n, n], dtype=np.int16)
    mask = np.zeros(sum_pixel.shape)
    mask[max_i:ones.shape[0] + max_i, max_j:ones.shape[1] + max_j] = ones
    temp = sum_pixel.copy()
    indices = (mask == 1) & (temp < 700)
    temp[indices] = 500
    ax2[1].axis('off')
    ax2[1].set_title('Mask')
    cax = ax2[1].imshow(temp)
    fig2.colorbar(cax, ax=ax2[1])
    fig2.savefig(OUTPUT_PATH + '/%s_%s_diff' % (patient, i), dpi=400)
    # plt.show()



