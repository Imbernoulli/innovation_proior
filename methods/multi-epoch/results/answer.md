# Multi-Epoch Finite-Sample Training For Sparse Parity

Multi-epoch finite-sample training for sparse parity keeps the two-layer MLP and fixed optimizer harness unchanged. It changes the resource regime: draw a small fixed training set once, repeatedly revisit it under the fixed `max_steps` budget, and keep weight decay on so the sparse parity solution can beat dense memorization.

## Problem

Learn `y = chi_S(x)` for a hidden `S subset [n]`, `|S| = k`. The statistical floor is `Theta(k log n)` labels, but correlational learners obey the SQ tradeoff

```text
T / tau^2 >= Omega(n^k).
```

Online single-pass SGD spends that computational cost in fresh examples because `T` steps with batch size `B` consume `T * B` independent samples. The finite-sample regime targets `k log n << m << n^k`: enough examples to identify `S` in principle, far too few to fund the online run.

## Key Idea

- The sparse signal is a Fourier-gap signal in the population gradient. For majority, with `n >= 4k`,
  `gamma_Maj >= 0.03 (n-1)^(-(k-1)/2) = Theta(n^(-(k-1)/2))`, and
  `|xi_{k-1}| = ((n-k)/(k-1)) |xi_{k+1}|`. A ReLU threshold indicator inherits this gap up to a constant factor.
- If a gradient estimate is within `gamma/2` in `infinity` norm for the applicable gradient-function gap `gamma`, the top `k` absolute gradient coordinates recover `S`. Bounded gradients need `O(log n / gamma^2)` samples for that one-shot condition.
- Reusing a dataset does not create new samples. It trades compute for samples only when the fixed empirical distribution already carries enough trace of `S`, and the optimizer can integrate that trace over many passes.
- Finite-data optimization introduces a dense memorizer. Weight decay supplies the low-norm bias toward the sparse rule, but only in a window: too little decay leaves memorization dominant; too much decay suppresses the tiny early sparse signal.
- In the fixed harness, the epoch count is determined by dataset size:

```text
epochs ~= max_steps * B / m.
```

The method is therefore: choose `m` small relative to the maximal one-pass budget, keep `m` above the statistical floor, reshuffle each pass, and keep weight decay in the working window.

## Final Form

Given the fixed model `Linear(n, W) -> ReLU -> Linear(W, 1) -> Sigmoid`, binary parity labels, batch size `B`, and a trainer that loops until `max_steps`:

1. Draw a fixed training set with `m = min(10_000, max_train_examples)`.
2. Reuse that same set for the whole run; the fixed trainer reshuffles it each pass.
3. Keep AdamW-style weight decay on, using the benchmark's standard base optimizer settings.

## Working Code

```python
import torch
from torch import nn


def parity_labels(x, secret):
    idx = torch.as_tensor(secret, dtype=torch.long, device=x.device)
    return x.index_select(1, idx).sum(dim=1).remainder(2).to(torch.float32)


def build_model(n_features, width):
    return nn.Sequential(
        nn.Linear(n_features, width),
        nn.ReLU(),
        nn.Linear(width, 1),
        nn.Sigmoid(),
    )


def make_dataset(secret, n_features, seed, max_train_examples):
    train_examples = 10_000
    m = min(train_examples, max_train_examples)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (m, n_features), generator=generator, dtype=torch.float32)
    y = parity_labels(x, secret)
    return x, y


def get_optimizer_config():
    return {"lr": 1e-3, "wd": 1e-2, "beta1": 0.9, "beta2": 0.999}


def train(model, train_x, train_y, optimizer, loss_fn, batch_size, max_steps):
    steps = 0
    while steps < max_steps:
        perm = torch.randperm(train_x.shape[0], device=train_x.device)
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[idx]).view(-1), train_y[idx])
            loss.backward()
            optimizer.step()
            steps += 1
            if steps >= max_steps:
                return
```

## Relation To Alternatives

- Online single-pass SGD uses fresh batches and keeps gradients unbiased for the population objective, but spends `n^{Omega(k)}` fresh samples.
- One-shot gradient recovery is mathematically clean, but needs `O(log n / gamma^2)` fresh samples for the `gamma/2` infinity-norm condition, and the full MLP theorem uses an `n^{Omega(k)}`-scale batch.
- Plain finite-data ERM with no effective weight decay can fit the sample with a dense memorizer. The multi-epoch finite-sample method keeps the finite data, adds repeated reshuffled passes, and retains the weight-decay bias toward the sparse rule.
