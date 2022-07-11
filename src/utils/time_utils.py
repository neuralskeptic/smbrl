import time


def timestamp():
    "human readable, sortable, no spaces, LOCAL TIME"
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
