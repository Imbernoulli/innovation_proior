## Research Question

A standard transformer encoder of `L` layers carries out `L` stages of computation and then
stops. The open question is whether an attention network with fixed weights and fixed depth can
be made to execute computations whose length is not known when the network is designed.

## Existing Tools

The transformer layer under discussion takes an input matrix `X in R^{d x n}`, with columns as
tokens, and applies multi-head attention followed by a position-wise ReLU network:

```text
Attn(X) = X + sum_h V_h X softmax(X^T K_h^T Q_h X)
f(X)    = Attn(X) + W2 ReLU(W1 Attn(X) + b1 1^T) + b2 1^T.
```

The softmax is column-wise and may use a temperature parameter. This gives two reusable facts.
First, attention can compare one column with all other columns through a bilinear score. Second,
a high-temperature softmax can approximate a hard selector when one score is separated from the
others by a margin. The feedforward sublayer supplies exact piecewise-linear gates and binary
arithmetic on bounded discrete encodings.

Several older ideas are relevant before any new construction is chosen. Binary encodings give a
low-width way to represent addresses and counters. One-instruction-set computers show that a
minimal instruction with subtraction and a conditional jump can be computationally universal.
Barron's approximation theorem says that bounded functions in a Fourier-integral class can be
approximated by sums of sigmoids with `O(m^{-1/2})` error.

## Baselines

RASP and Tracr show that some sequence algorithms can be compiled into transformer weights.
They make the programming interface explicit, with a compiled transformer that grows with the
program. The language is not a general low-level machine model.

In-context learning constructions by Akyurek et al. and von Oswald et al. show hardcoded
transformers that implement gradient-descent-like updates. These are concrete and algebraic,
covering a specific learning problem with a fixed number of update steps per forward pass.

Turing-completeness results for transformers prove broad expressivity. They establish that
attention architectures can simulate computation under appropriate assumptions.

Universal Transformers and other recurrent attention architectures reuse a block over multiple
steps, showing that recurrence around attention is a reasonable architectural move.

## Evaluation Frame

The right checks are structural. One should ask how many transformer layers, heads, and width
coordinates are required for one computational step; whether those quantities remain fixed as
the number of executed steps grows; how state is addressed and overwritten; how branch
conditions are represented; and how softmax approximation error is kept below a requested
tolerance.

For a useful construction, the input length may grow with the amount of memory and program text,
and the number of invocations may grow with runtime. Any claim of exact or approximate execution
has to specify the numeric encoding, the valid integer range, the softmax temperature or padding
constants that control error, and the cleanup operation that prevents small errors from
accumulating indefinitely.

## Starting Scaffold

The scaffold available before the key design decision is only a fixed transformer block and an
external driver that can call it repeatedly.

```python
def transformer_layer(X, Qs, Ks, Vs, W1, b1, W2, b2, lam):
    attn = X
    for Q, K, V in zip(Qs, Ks, Vs):
        scores = X.T @ K.T @ Q @ X
        P = column_softmax(lam * scores)
        attn = attn + V @ X @ P
    return attn + W2 @ relu(W1 @ attn + b1) + b2


def run_recurrent_block(block, X, T):
    for _ in range(T):
        X = block(X)
    return X
```
