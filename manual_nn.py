import numpy as np

class ManualMLP:
    """Small 2-layer MLP with manual backpropagation.

    Architecture:
        X -> Linear(W1,b1) -> tanh -> Linear(W2,b2) -> sigmoid

    Binary cross-entropy loss is used. No autograd/backward is used
    anywhere in the implementation.
    """

    def __init__(self, input_dim=2, hidden_dim=8, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros((1, 1))
        self.cache = None

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = self.sigmoid(z2)
        self.cache = (X, z1, a1, z2, y_hat)
        return y_hat

    @staticmethod
    def loss(y_hat, y):
        eps = 1e-8
        return -np.mean(y * np.log(y_hat + eps) +
                        (1 - y) * np.log(1 - y_hat + eps))

    def backward(self, y):
        X, z1, a1, z2, y_hat = self.cache
        n = X.shape[0]

        # BCE + sigmoid simplifies to:
        # dL/dz2 = (y_hat - y) / n
        dz2 = (y_hat - y) / n
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - np.tanh(z1) ** 2)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads, lr=0.05):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
