import os
import sys


def loadDataStore(ds_path):
    data_lines = []
    with open(ds_path, "r") as f:
        data_lines.append(f.readline())
    f.closed

    print(data_lines)
