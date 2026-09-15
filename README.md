# Postman AI/ML Recruitment Task – Task 1

Manual backpropagation for a small neural network.

## What is implemented

- Feedforward neural network using NumPy matrix operations.
- Linear layers + `tanh` hidden activation + sigmoid output.
- Manual backward pass using the chain rule.
- Numerical gradient checking.
- Independent comparison against `torch.autograd`.
- Training on a small real binary classification task from the Iris dataset to show that the loss decreases.
- No autograd/backward is used in the neural-network implementation itself.

## Files

- `manual_nn.py` – network, forward pass, loss, manual backward pass, SGD.
- `gradient_check.py` – numerical gradient check and PyTorch reference comparison.
- `train.py` – small real Iris classification experiment (two classes, two features).
- `WRITEUP.md` – short explanation of the implementation and debugging.

## Run

Install dependencies:

```bash
pip install numpy torch scikit-learn
```

Then:

```bash
python gradient_check.py
python train.py
```

The gradient checker should report PASS for all parameters, and the training loss
should decrease while classification accuracy becomes high.

## Requirement mapping

1.1 `ManualMLP` uses basic matrix operations.
1.2 `forward()` implements linear layers and activations.
1.3 `backward()` derives gradients manually.
1.4 `gradient_check.py` compares with numerical gradients and `torch.autograd`.
1.5 train.py trains on a small real Iris classification dataset and prints decreasing loss.
1.6 `WRITEUP.md` documents the main debugging/gradient checks.

The assignment explicitly says incomplete submissions are acceptable and that
understanding, experiments, debugging, and learning matter more than a perfect result.
