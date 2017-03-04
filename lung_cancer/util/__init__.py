# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement
import os

PATH = os.path.dirname(__file__)
PATH = os.path.dirname(PATH)
INPUT_PATH = PATH + '/data/stage1'
OUTPUT_PATH = PATH + '/data'
PROCESS_NUM = 12


def listdir_no_hidden(path):
    #  Use sorted for more predicative behaviour
    return sorted([f for f in os.listdir(path) if not f.startswith('.')])
