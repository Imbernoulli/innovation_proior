# Design a Regularizer for an Overfit-Prone Random-Feature Net

## Background

You are handed a small regression model that is **badly overfit-prone**. The model is a
wide **random Fourier feature (RFF)** net: the input `x` (a scalar) is mapped through `M = 200`
fixed random features

```
phi_j(x) = scale * cos(omega_j * x + phase_j),      j = 1..M
```

and the prediction is linear in those features, `y_hat = phi(x) . w`. The frequencies
`omega_j` are drawn from a broad band, so there are plenty of **high-frequency features** that
can memorize label noise. With only **25 noisy training points**, a plain fit drives the
training error to ~0 while the held-out error blows up: a textbook generalization gap.

The **trainer is fixed and out of your control**: full-batch gradient descent for `T = 8000`
steps at learning rate `lr`, starting from `w = 0`, minimizing

```
(1/n) * sum_i (phi(x_i).w - y_i)^2   +   REGULARIZER(w)
```

**Your job is to design the REGULARIZER** — nothing else. A good regularizer must suppress the
noise-fitting capacity without destroying the signal, closing the generalization gap.

## The regularizer you may design

Your answer specifies a penalty of the form

```
REGULARIZER(w) = sum_j ridge_j * w_j^2   +   l1 * sum_j |w_j|
```

optionally trained with **input jitter** (Gaussian noise of std `jitter` added to the training
inputs each step — data augmentation) and **decoupled weight decay** (`w <- w * (1 - lr*weight_decay)`
after each step). The trainer applies exactly what you specify; everything is deterministic.

You are free to make `ridge_j` **depend on the feature** — e.g. penalize high-frequency features
more than low-frequency ones. The public instance gives you every `omega_j` and `phase_j`, the
training data, and the training seed, so you can reason about (or even internally simulate) the
trainer.

## Input (public instance, one JSON object on stdin)

```
{
  "M": 200,                 # number of random features
  "T": 8000,                # fixed GD steps
  "lr": 0.03,               # fixed learning rate
  "scale": <float>,         # feature scale sqrt(2/M)
  "omega": [<M floats>],    # feature frequencies
  "phase": [<M floats>],    # feature phases
  "xtr":   [<25 floats>],   # training inputs
  "ytr":   [<25 floats>],   # training targets (noisy)
  "train_seed": <int>,      # RNG seed the trainer uses (for jitter)
  "caps": {"ridge":20.0, "l1":100.0, "jitter":3.0, "weight_decay":20.0}
}
```

The held-out test set is **hidden**; it is never sent to your program.

## Output (one JSON object on stdout)

```
{
  "ridge": <number OR list of M numbers>,   # per-feature L2 weight(s), each in [0, 20]
  "l1": <number>,            # optional, in [0, 100],  default 0
  "jitter": <number>,        # optional, in [0, 3],    default 0
  "weight_decay": <number>   # optional, in [0, 20],   default 0
}
```

- `ridge` may be a single number (uniform) or a length-`M` list (per-feature). Omit it for zero.
- All values must be **finite and non-negative** and within the caps above. Any `NaN`/`inf`,
  negative value, out-of-range value, or wrong-length `ridge` list makes the answer **infeasible**
  (scored 0 on that instance).

## Objective — MINIMIZE

The evaluator trains the fixed net with your regularizer and measures the **held-out
mean-squared error** (the generalization gap made concrete). Lower is better.

## Scoring

For each of `10` instances (varying noise level and feature bandwidth), let `B` be the held-out
MSE with **no** regularization (the overfit baseline the evaluator computes itself) and `obj` the
held-out MSE with your regularizer. The per-instance score is

```
r = min(1.0, 0.1 * B / obj)
```

so doing nothing scores ~`0.1`, and halving/quartering the held-out error scores proportionally
higher. The reported `Ratio` is the mean over all instances. There is an irreducible noise floor,
so no regularizer reaches `1.0` — real headroom, multiple viable strategies (uniform ridge, L1,
input jitter, or a frequency-weighted smoothness prior), and no closed-form optimum.

Your program is run in an isolated sandbox and only ever sees the public instance.
