# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import lung_cancer
from lung_cancer import listdir_no_hidden
import preprocess as pre
import audit

PATH = lung_cancer.PATH


def main():
    input_folder = PATH + '/data/stage1/'
    patients = listdir_no_hidden(input_folder)
    patients.sort()

    # audit.folder_name_equals_id()

    # first_path = input_folder + patients[0]
    # first_slices = pre.load_slices(first_path)
    # first_pixel = pre.slices_to_pixel(first_slices)
    # first_pixel_resampled, new_spacing = pre.resample(first_pixel, first_slices, [5, 5, 5])
    # pre.plot_3d(first_pixel, 400)
    # pre.plot_3d(first_pixel_resampled, 400)
    # pre.plot_3d(first_pixel, 0)
    # pre.plot_3d(first_pixel_resampled, 0)

    # first_pixel = pre.load_pixel(first_path)
    # print(first_pixel.shape)

    # for patient in patients:
    #     path = input_folder + patient
    #     pixel = pre.load_pixel(path)
    #     print(pixel.shape)


if __name__ == "__main__":
    main()
