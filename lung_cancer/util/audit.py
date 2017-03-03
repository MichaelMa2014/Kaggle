# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import dicom

from util import PATH, listdir_no_hidden

DATA_PATH = PATH + '/data/stage1'


def folder_name_equals_id():
    for patient in listdir_no_hidden(DATA_PATH):
        # use 1 because 0 might be .npy
        first_path = DATA_PATH + '/' + patient + '/' + listdir_no_hidden(DATA_PATH + '/' + patient)[1]
        first_slice = dicom.read_file(first_path)
        if not patient == first_slice.PatientID:
            print('Folder name: %s\nPatient ID:%s\n\n' % (patient, first_slice.PatientID))
            return False
    print('AUDITING folder name equals patient ID')
