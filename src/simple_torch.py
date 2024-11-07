import os
import sys


def loadDataStore(ds_path):
    data_lines = []
    print(f"ds_path: {ds_path}")
    with open(ds_path, "r", encoding="utf-8") as f:
        data_lines.append(f.readline())
    f.close()

    print(data_lines)
