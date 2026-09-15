import numpy as np
import torch
from manual_nn import ManualMLP

def numerical_gradient(model, X, y, name, eps=1e-5):
    param = getattr(model, name)
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        old = param[idx]

        param[idx] = old + eps
        plus = model.loss(model.forward(X), y)

        param[idx] = old - eps
        minus = model.loss(model.forward(X), y)

        param[idx] = old
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()

    return grad

def main():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(5, 2))
    y = rng.integers(0, 2, size=(5, 1)).astype(float)

    model = ManualMLP(input_dim=2, hidden_dim=4, seed=3)
    model.forward(X)
    manual = model.backward(y)

    print("Numerical gradient check:")
    for name in ["W1", "b1", "W2", "b2"]:
        num = numerical_gradient(model, X, y, name)
        rel_err = np.linalg.norm(manual[name] - num) / (
            np.linalg.norm(manual[name]) + np.linalg.norm(num) + 1e-12
        )
        print(f"{name}: relative error = {rel_err:.3e} -> "
              f"{'PASS' if rel_err < 1e-6 else 'FAIL'}")

    # Independent torch reference using the same parameters.
    dtype = torch.float64
    tX = torch.tensor(X, dtype=dtype)
    ty = torch.tensor(y, dtype=dtype)

    tW1 = torch.tensor(model.W1, dtype=dtype, requires_grad=True)
    tb1 = torch.tensor(model.b1, dtype=dtype, requires_grad=True)
    tW2 = torch.tensor(model.W2, dtype=dtype, requires_grad=True)
    tb2 = torch.tensor(model.b2, dtype=dtype, requires_grad=True)

    tz1 = tX @ tW1 + tb1
    ta1 = torch.tanh(tz1)
    tz2 = ta1 @ tW2 + tb2
    tyhat = torch.sigmoid(tz2)
    tloss = -(ty * torch.log(tyhat + 1e-8) +
              (1 - ty) * torch.log(1 - tyhat + 1e-8)).mean()
    tloss.backward()

    torch_grads = {
        "W1": tW1.grad.detach().numpy(),
        "b1": tb1.grad.detach().numpy(),
        "W2": tW2.grad.detach().numpy(),
        "b2": tb2.grad.detach().numpy(),
    }

    print("\nComparison with torch.autograd reference:")
    for name in manual:
        diff = np.max(np.abs(manual[name] - torch_grads[name]))
        print(f"{name}: max abs diff = {diff:.3e} -> "
              f"{'PASS' if diff < 1e-6 else 'FAIL'}")

if __name__ == "__main__":
    main()
