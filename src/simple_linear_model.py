#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SimpleLinerModel 简单线性模型"""

import numpy as np
import matplotlib.pyplot as plt


class SimpleLinearModel:
    """_summary_"""

    def __init__(self, x, y):
        print(f"x: {x}, y: {y}")  # 调试输出
        self.w = np.random.randn()
        self.b = np.random.randn()
        self.x = x
        self.y = y
        self.losses = []

    def predict(self, c):
        """预测"""
        return self.w * c + self.b

    def mean_squared_error(self, yt: np.array, yp: np.array):
        return ((yt - yp) ** 2).mean()

    def gradient_descent(self, learning_rate=0.01, epochs=10000):
        for epoch in range(epochs):
            yp = self.predict(self.x)  # 计算预测值
            loss = self.mean_squared_error(self.y, yp)  # 计算损失
            self.losses.append(loss)  # 保存当前损失
            # 计算梯度
            dw = -2 * np.dot(self.x, (self.y - yp)) / len(self.x)
            db = -2 * np.sum(self.y - yp) / len(self.x)
            # 更新参数
            self.w -= learning_rate * dw
            self.b -= learning_rate * db
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")

    def plot_loss(self):
        plt.plot(range(len(self.losses)), self.losses, label="losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.show()


def test_simple_liner_model():
    print("start testSimpleLinerModel")
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    model = SimpleLinearModel(x, y)
    model.gradient_descent()
    model.plot_loss()
