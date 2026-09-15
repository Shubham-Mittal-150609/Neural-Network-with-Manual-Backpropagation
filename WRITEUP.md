# Write-up: Manual Backpropagation

## 1. Goal

The goal of this project was to build a small feedforward neural network and
implement backpropagation manually using basic matrix operations.

The network was implemented using NumPy without using autograd or `.backward()`
inside the neural-network implementation.

The model architecture is:

`Input -> Linear Layer -> tanh -> Linear Layer -> sigmoid`

Binary cross-entropy was used as the loss function because the training task
is a binary classification problem.

The main objectives were to understand the forward pass, derive the gradients
using the chain rule, verify the gradients, and train the network on a small
real dataset.

---

## 2. Model Architecture

The network contains two linear layers and one hidden activation function.

For an input batch `X`, the first linear layer is:

`Z1 = X W1 + b1`

The hidden activation is:

`A1 = tanh(Z1)`

The second linear layer is:

`Z2 = A1 W2 + b2`

Finally, the sigmoid activation produces the predicted probability:

`Yhat = sigmoid(Z2)`

The model uses a hidden layer with 8 neurons.

The parameters of the network are:

- `W1`: weights of the first layer
- `b1`: bias of the first layer
- `W2`: weights of the second layer
- `b2`: bias of the second layer

All of these parameters are updated manually using gradient descent.

---

## 3. Forward Pass and Loss

The forward pass is implemented using NumPy matrix multiplication.

The binary cross-entropy loss used for training is:

`L = -mean(y log(Yhat) + (1-y) log(1-Yhat))`

The loss measures the difference between the predicted probabilities and the
true binary labels.

For numerical stability, a small epsilon is added inside the logarithms.

The sigmoid function is:

`sigmoid(z) = 1 / (1 + exp(-z))`

The derivative of the sigmoid is:

`d sigmoid(z) / dz = sigmoid(z)(1 - sigmoid(z))`

---

## 4. Manual Backward Pass

The backward pass was derived using the chain rule.

For the combination of sigmoid activation and binary cross-entropy loss, the
gradient simplifies to:

`dL/dZ2 = (Yhat - y) / N`

where `N` is the number of samples in the batch.

The gradients for the second linear layer are:

`dL/dW2 = A1^T dL/dZ2`

`dL/db2 = sum(dL/dZ2)`

The gradient flowing back into the hidden activation is:

`dL/dA1 = dL/dZ2 W2^T`

For the tanh activation:

`d tanh(Z1) / dZ1 = 1 - tanh(Z1)^2`

Therefore:

`dL/dZ1 = dL/dA1 * (1 - tanh(Z1)^2)`

The gradients for the first linear layer are:

`dL/dW1 = X^T dL/dZ1`

`dL/db1 = sum(dL/dZ1)`

After calculating all gradients, the parameters are updated using simple
stochastic gradient descent:

`W = W - learning_rate * dL/dW`

The same update rule is applied to the biases.

---

## 5. Gradient Verification

Gradient verification was an important part of the implementation.

Two different methods were used to check the manually calculated gradients.

### 5.1 Numerical Gradient Checking

For each parameter, the numerical gradient was estimated using the central
difference formula:

`(L(theta + eps) - L(theta - eps)) / (2 eps)`

The numerical gradients were then compared with the gradients produced by the
manual backward pass.

The results were:

```text
W1: relative error = 5.922e-08 -> PASS
b1: relative error = 6.274e-08 -> PASS
W2: relative error = 5.267e-08 -> PASS
b2: relative error = 5.495e-08 -> PASS
```

All four parameter gradients passed the numerical gradient check with very
small relative errors.

### 5.2 PyTorch Reference

I also implemented the same mathematical computation in PyTorch and used
`torch.autograd` as a trusted reference.

Autograd was used only for verification and not for the actual neural-network
implementation.

The maximum absolute differences were:

```text
W1: max abs diff = 2.338e-08 -> PASS
b1: max abs diff = 2.217e-08 -> PASS
W2: max abs diff = 3.233e-08 -> PASS
b2: max abs diff = 3.251e-08 -> PASS
```

The very small differences show that the manually calculated gradients closely
match the reference implementation.

---

## 6. Training Experiment

For the training experiment, I used a small real binary classification task
from the Iris dataset.

I selected the first two classes from the dataset and used two input features:
sepal length and sepal width.

This keeps the experiment small while still using a real dataset.

The input features were standardized before training.

Since the task has two classes, sigmoid activation with binary cross-entropy
is suitable for the output layer.

The network was trained for 2000 epochs using a learning rate of 0.1.

The training results were:

```text
Epoch 1:
Loss     = 0.649952
Accuracy = 0.590

Epoch 200:
Loss     = 0.037099
Accuracy = 1.000

Epoch 1000:
Loss     = 0.006885
Accuracy = 1.000

Epoch 2000:
Loss     = 0.003442
Accuracy = 1.000
```

The loss decreased from `0.649952` to `0.003442`.

The final training accuracy was `100%`.

This demonstrates that the manually implemented network was able to learn the
classification task successfully.

---

## 7. Debugging and Mistakes

I did not encounter a major gradient bug in the final implementation, but
gradient checking was the main debugging technique used to verify the backward
pass.

Each parameter was checked separately using numerical differentiation. The
manual gradients were then compared with the PyTorch reference implementation.

This made it possible to verify individual parts of the backward pass instead
of relying only on the final training accuracy.

In particular, the checks helped confirm that the batch-size factor `1/N` in
the loss gradient was handled correctly and that the derivative of the tanh
activation was implemented correctly.

The very small gradient errors provided confidence that the chain-rule
implementation was correct before using it for training.

---

## 8. What I Learned

This project helped me understand that backpropagation is essentially repeated
application of the chain rule.

One important practical lesson was that the backward pass needs intermediate
values from the forward pass. For example, the hidden activation `A1` is needed
to calculate the gradient of `W2`.

Another important lesson was the value of gradient checking. A model can appear
to train even when some gradients are implemented incorrectly, so checking the
gradients independently provides a much stronger validation of the
implementation.

Comparing the manual implementation against both numerical gradients and a
trusted PyTorch reference gave a clear way to identify whether the mathematical
derivatives were correct.

Overall, implementing the network manually made the relationship between the
forward pass, loss function, chain rule, gradients, and parameter updates much
clearer than using an automatic differentiation framework alone.
