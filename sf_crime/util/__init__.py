import os
import inspect

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(ROOT_PATH, "data")
OUTPUT_PATH = os.path.join(ROOT_PATH, "data")


def print_name_and_value(var):
    frame = inspect.stack()[1]
    # list(frame) is [<frame object at 0x7f69511bf828>, '__init__.py', 20, '<module>', ['print_name_and_value(a)\n'], 0]
    lines = frame[4]
    var_name = ''.join(lines).strip().split('(')[-1].split(')')[0]
    print("%s: %d" % (var_name, var))

