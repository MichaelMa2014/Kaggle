# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import data


def main():
    input_folder = './data/stage1/'
    patients = os.listdir(input_folder)
    patients.sort()
    first_patient_slices = data.load_slices(input_folder + patients[1])
    first_patient_pixel = data.slices_to_pixel(first_patient_slices)
    data.plot_3d(first_patient_pixel, 400)
    first_patient_pixel_resampled, new_spacing = data.resample(first_patient_pixel, first_patient_slices, [5, 5, 5])
    data.plot_3d(first_patient_pixel_resampled, 400)


if __name__ == '__main__':
    main()
