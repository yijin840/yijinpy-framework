#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main func"""
import sys
import os

abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(abspath)
sys.path.append(abspath)
sys.path.append(abspath + "/src/test")

from yijin_model import YijinGptModel
import file_utils
import test.torch_test as torch_test
import simple_torch
import test.simple_linear_model as simple_linear_model
import test.numpy_utils as numpy_utils


def test():
    print("test test")
    # numpy_arr()
    # testSimpleLinerModel()
    torch_test.run()


def load_model_and_eval(model_path, a, b, c, d):
    torch_test.load_model_and_eval(model_path, a, b, c, d)


def simple_torch_test():
    current_working_dir = os.getcwd()
    simple_torch.load_data_store(
        current_working_dir + "/yijinpy-framework/resources/WikiQACorpus/WikiQA-dev.txt"
    )


def serilable_file(file_path):
    return file_utils.serilable_file(file_path)


def load_serilable_data(data):
    return file_utils.load_serilable_data(data)


def yijin_model_test():
    model = YijinGptModel()
    # model.train()
    prompt = "你是谁啊?"
    response = model.generate_response(prompt)
    print(response)
    # print("hello world")


if __name__ == "__main__":
    print("hello world")
