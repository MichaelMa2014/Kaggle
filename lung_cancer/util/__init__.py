# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os
import pandas as pd

PATH = os.path.dirname(__file__)
PATH = os.path.dirname(PATH)
INPUT_PATH = PATH + '/data/stage1'
OUTPUT_PATH = PATH + '/data'
PROCESS_NUM = 1

labels = dict()


def listdir_no_hidden(path):
    #  Use sorted for more predicative behaviour
    return sorted([f for f in os.listdir(path) if not f.startswith('.')])


def load_labels():
    label_df = pd.read_csv(PATH + '/data/stage1_labels.csv')
    patients = listdir_no_hidden(INPUT_PATH)
    patients.sort()
    for patient in patients:
        try:
            label = label_df[label_df.id == patient].iloc[0, 1]
            labels[patient] = label
        except IndexError:
            labels[patient] = None
            pass

load_labels()
