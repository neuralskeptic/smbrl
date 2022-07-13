import time
from datetime import datetime


def timestamp():
    "human readable, sortable, no spaces, LOCAL TIME"
    # return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
