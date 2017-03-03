# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import dicom

from util import INPUT_PATH, listdir_no_hidden
import util.preprocess as pre


def folder_name_equals_id():
    for patient in listdir_no_hidden(INPUT_PATH):
        # use 1 because 0 might be .npy
        first_path = INPUT_PATH + '/' + patient + '/' + listdir_no_hidden(INPUT_PATH + '/' + patient)[1]
        first_slice = dicom.read_file(first_path)
        if not patient == first_slice.PatientID:
            print('Folder name: %s\nPatient ID:%s\n\n' % (patient, first_slice.PatientID))
            return False
    print('AUDITING folder name equals patient ID')


def slice_size_is_512_by_512():
    for patient in listdir_no_hidden(INPUT_PATH):
        path = INPUT_PATH + '/' + patient
        pixel = pre.load_pixel(path)
        if not pixel.shape[1] == 512 or not pixel.shape[2] == 512:
            print('%s slice size' % patient, pixel.shape)
            return False
    print('AUDITING slice size is always 512 * 512')
