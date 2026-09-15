# Write-up: Manual Backpropagation

## 1. Goal

I implemented a small multilayer perceptron from scratch with NumPy. The model is:

`X -> Linear -> tanh -> Linear -> sigmoid`

The loss is binary cross-entropy. The neural-network gradients are calculated
manually; the implementation does not call autograd or `.backward()`.

## 2. Forward pass

For a batch `X`:

`Z1 = X W1 + b1`

`A1 = tanh(Z1)`

`Z2 = A1 W2 + b2`

`Yhat = sigmoid(Z2)`

Binary cross-entropy is:

`L = -mean(y log(Yhat) + (1-y) log(1-Yhat))`

## 3. Backward pass

For sigmoid + binary cross-entropy:

`dL/dZ2 = (Yhat - y) / N`

Then:

`dL/dW2 = A1^T dL/dZ2`

`dL/db2 = sum(dL/dZ2)`

`dL/dA1 = dL/dZ2 W2^T`

For tanh:

`d tanh(Z1) / dZ1 = 1 - tanh(Z1)^2`

Therefore:

`dL/dZ1 = dL/dA1 * (1 - tanh(Z1)^2)`

Finally:

`dL/dW1 = X^T dL/dZ1`

`dL/db1 = sum(dL/dZ1)`

The parameter update is simple SGD:

`W <- W - learning_rate * dL/dW`

## 4. Gradient verification

I used two checks.

First, numerical differentiation estimates each derivative with:

`(L(theta + eps) - L(theta - eps)) / (2 eps)`

Second, I constructed the same computation in PyTorch and used
`torch.autograd` only as a trusted reference.

The relative errors/max absolute differences should be very small, which
supports that the manual chain-rule implementation is correct.

## 5. Training experiment

I used a small real binary classification task from the Iris dataset. I kept
the first two classes and two input features (sepal length and sepal width)
so that the experiment remains small and easy to inspect.

The labels are binary, so sigmoid + binary cross-entropy is appropriate.
The input features are standardized before training. During training, the
loss decreases and the classification accuracy becomes high.

## 6. Debugging / mistakes

The most useful debugging technique was gradient checking. A common source of
errors in this model is forgetting the batch-size factor `1/N` in the
binary cross-entropy gradient, or using the wrong derivative for `tanh`.

Checking every parameter separately makes these mistakes easy to localize:
if only `W1`/`b1` fail, the problem is likely in the hidden-layer chain rule;
if `W2`/`b2` fail, the output-layer derivative is the first place to inspect.

## 7. What I learned

Backpropagation is repeated application of the chain rule. The important
practical lesson is that every intermediate value needed by the backward pass
must be retained from the forward pass. Numerical gradient checking is a
simple but powerful way to verify an implementation before trusting training
results.
