import numpy as np

class SimpleModel:
    def __init__(self, input_size, lr=0.01):
        self.weights = np.random.randn(input_size)
        self.lr = lr

    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            preds = X.dot(self.weights)
            loss = np.mean((preds - y) ** 2)
            grad = 2 * X.T.dot(preds - y) / len(X)
            self.weights -= self.lr * grad
            print(f"Loss: {loss}")

X_train = np.random.randn(100, 1)
y_train = 3 * X_train + 2 + np.random.randn(100, 1) * 0.1
model = SimpleModel(1)
model.train(X_train, y_train)
