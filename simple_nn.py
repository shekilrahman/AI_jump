import numpy as np
import os

class SimpleNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(hidden_size1, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size1, 1))
        self.W2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2.0 / hidden_size1)
        self.b2 = np.zeros((hidden_size2, 1))
        self.W3 = np.random.randn(output_size, hidden_size2) * np.sqrt(2.0 / hidden_size2)
        self.b3 = np.zeros((output_size, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1.0 - s)

    def forward(self, x):
        self.z1 = self.W1.dot(x) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.W2.dot(self.a1) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.W3.dot(self.a2) + self.b3
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    def backward(self, x, y_true):
        m = 1 
        dz3 = (self.a3 - y_true) * self.sigmoid_derivative(self.z3)
        dw3 = dz3.dot(self.a2.T)
        db3 = dz3

        da2 = self.W3.T.dot(dz3)
        dz2 = da2 * self.relu_derivative(self.z2)
        dw2 = dz2.dot(self.a1.T)
        db2 = dz2

        da1 = self.W2.T.dot(dz2)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = dz1.dot(x.T)
        db1 = dz1

        self.W3 -= self.lr * dw3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def train(self, x, y_true):
        self.forward(x)
        self.backward(x, y_true)

    def save(self, filepath):
        np.savez(filepath,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3,
                 lr=self.lr)

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No NN file at '{filepath}'")
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        if 'lr' in data:
            self.lr = float(data['lr']) 
