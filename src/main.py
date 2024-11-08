#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main func"""

import os
from t_model import TModel
import numpy_utils
import simple_linear_model
import simple_torch
import torch_utils
import file_utils


def run():
    """run"""
    print("hello world")
    TModel.load()
    TModel.train()


def test():
    print("test test")
    # numpy_arr()
    # testSimpleLinerModel()
    torch_utils.run()


def loadModelAndEval(model_path, a, b, c, d):
    torch_utils.loadModelAndEval(model_path, a, b, c, d)


def simple_torch_test():
    current_working_dir = os.getcwd()
    simple_torch.loadDataStore(
        current_working_dir + "/yijinpy-framework/resources/WikiQACorpus/WikiQA-dev.txt"
    )


def serilableFile(file_path):
    return file_utils.serilableFile(file_path)


def loadSerilableData(data):
    return file_utils.loadSerilableData(data)


if __name__ == "__main__":
    print("hello world")
    run()
    test()
