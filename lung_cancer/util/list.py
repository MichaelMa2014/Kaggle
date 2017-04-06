# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

from util import INPUT_PATH, listdir_no_hidden, labels


def pos_list():
    """
    Return a sorted list of all postive patients
    """
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()

    l = []
    for patient in patients:
        if labels[patient] == 1:
            l.append(patient)

    return l


def neg_list():
    """
    Return a sorted list of all negative patients
    """
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()

    l = []
    for patient in patients:
        if labels[patient] == 0:
            l.append(patient)

    return l


def test_list():
    """
    Return a sorted list of all test patients
    """
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()

    l = []
    for patient in patients:
        if labels[patient] == None:
            l.append(patient)

    return l


def train_list():
    """
    Return a sorted list of all train patients
    """
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()

    l = []
    for patient in patients:
        if labels[patient] != None:
            l.append(patient)

    return l
