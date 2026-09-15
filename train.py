import numpy as np
from sklearn.datasets import load_iris
from manual_nn import ManualMLP

def load_small_real_dataset():
    """Use a small real binary classification task from the Iris dataset.

    We keep classes 0 and 1 (Setosa vs Versicolor) and two features so the
    experiment stays small and easy to understand.
    """
    iris = load_iris()
    mask = iris.target < 2
    X = iris.data[mask, :2].astype(float)  # sepal length, sepal width
    y = iris.target[mask].reshape(-1, 1).astype(float)

    # Standardize using the whole small dataset for a simple demonstration.
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    return X, y

def main():
    X, y = load_small_real_dataset()
    model = ManualMLP(input_dim=2, hidden_dim=8, seed=42)

    for epoch in range(1, 2001):
        yhat = model.forward(X)
        loss = model.loss(yhat, y)
        grads = model.backward(y)
        model.step(grads, lr=0.1)

        if epoch == 1 or epoch % 200 == 0:
            acc = np.mean((yhat >= 0.5) == y)
            print(f"epoch={epoch:4d} loss={loss:.6f} accuracy={acc:.3f}")

    print("\nSample predictions:")
    for i in range(10):
        pred = int(model.predict(X[i:i+1])[0, 0])
        print(f"sample={i:2d} true={int(y[i,0])} predicted={pred}")

if __name__ == "__main__":
    main()
