# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import time


def _INFO(statement):
    print("[INFO] " + str(time.time()) + " " + str(statement))


def _ERROR(statement):
    print("[ERROR] " + str(time.time()) + " " + str(statement))
