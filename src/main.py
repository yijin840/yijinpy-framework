#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main func"""

import os
from t_model import TModel
from numpy_utils import numpy_arr
from simple_linear_model import testSimpleLinerModel
from torch_utils import torch_start
from simple_torch import loadDataStore


def run():
    """run"""
    print("hello world")
    TModel.load()
    TModel.train()


def test():
    print("test test")
    # numpy_arr()
    # testSimpleLinerModel()
    torch_start()


def simple_torch_test():
    current_working_dir = os.getcwd()
    loadDataStore(current_working_dir + "/resources/WikiQACorpus/WikiQA-dev.txt")


if __name__ == "__main__":
    print("hello world")
    run()
    test()
