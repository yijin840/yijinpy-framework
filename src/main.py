#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main func"""

from t_model import TModel
from numpy_utils import numpy_arr


def run():
    """run"""
    print("hello world")
    TModel.load()
    TModel.train()


def test():
    print("test test")
    numpy_arr()


if __name__ == "__main__":
    print("hello world")
    run()