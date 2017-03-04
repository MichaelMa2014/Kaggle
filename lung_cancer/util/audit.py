# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import sys
import dicom

from util import INPUT_PATH, listdir_no_hidden
import util.preprocess as pre


def folder_name_equals_id():
    """
    Confirmed for stage1
    :return:
    """
    for patient in listdir_no_hidden(INPUT_PATH):
        # use 1 because 0 might be .npy
        first_path = INPUT_PATH + '/' + patient + '/' + listdir_no_hidden(INPUT_PATH + '/' + patient)[1]
        first_slice = dicom.read_file(first_path)
        if not patient == first_slice.PatientID:
            print('Folder name: %s\nPatient ID:%s\n\n' % (patient, first_slice.PatientID))
            return False
    print('AUDITING folder name equals patient ID')


def slice_size_is_512_by_512():
    """
    Confirmed for stage1
    :return:
    """
    for patient in listdir_no_hidden(INPUT_PATH):
        path = INPUT_PATH + '/' + patient
        pixel = pre.load_pixel(path)
        if not pixel.shape[1] == 512 or not pixel.shape[2] == 512:
            print('%s slice size' % patient, pixel.shape)
            return False
    print('AUDITING slice size is always 512 * 512')


def slope_and_intercept_consistent():
    consistent_all = True
    slope_all = None
    intercept_all = None
    for patient in listdir_no_hidden(INPUT_PATH):
        sys.stdout.flush()
        consistent = True
        path = INPUT_PATH + '/' + patient
        slices = pre.load_slices(path)
        slope = slices[0].RescaleSlope
        intercept = slices[0].RescaleIntercept
        if slope_all is None and intercept_all is None:
            print('%s is the first patient: %s, %s' % (patient, slope, intercept))
            slope_all = slope
            intercept_all = intercept
        elif not slope == slope_all or not intercept == intercept_all:
            print('%s not consistent with the first patient: %s, %s' % (patient, slope, intercept))
            consistent_all = False
        for i in range(len(slices)):
            if not slices[i].RescaleSlope == slope:
                print('%s slope (%s) not consistent in slice %s' % (patient, slices[i].RescaleSlope, i))
                consistent = False
            if not slices[i].RescaleIntercept == intercept:
                print('%s intercept (%s) not consistent in slice %s' % (patient, slices[i].RescaleIntercept, i))
                consistent = False
        if consistent:
            print('AUDITING slope and intercept consistent for %s' % patient)
    if consistent_all:
        print('AUDITING slope and intercept consistent')
    return consistent_all
