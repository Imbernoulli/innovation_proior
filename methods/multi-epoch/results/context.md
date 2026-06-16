## Research question

Fix the sparse-parity feature-learning testbed: inputs are drawn uniformly from the Boolean cube, and the label is the parity of an unknown `k`-subset of coordinates. In `±1` notation this is
`y = chi_S(x) = prod_{i in S} x_i`; in a binary `{0,1}` harness it is the sum of the selected bits modulo two. The learner knows `n` and `k`, but not `S`, and it must recover a classifier that generalizes on fresh Boolean-cube samples.

The problem separates statistical information from computation. There are `C(n,k)` possible hidden subsets, so the information-theoretic sample floor is `log C(n,k) = Theta(k log n)`. By contrast, correlational algorithms face the statistical-query lower bound: if an algorithm uses `T` queries at tolerance `tau`, then solving `(n,k)`-parity requires

```text
T / tau^2 >= Omega(n^k).
```

Gradient training falls into this correlational picture unless it obtains very accurate gradient estimates. The open regime is therefore

```text
k log n << m << n^k,
```

where a finite sample can contain enough information to specify `S`, but the usual fresh-batch training picture would need far more independent examples than are available.

## Background

Parities are orthogonal under the correlation inner product. For any guessed subset `S'`,

```text
E_x[chi_S(x) chi_{S'}(x)] = 1{S = S'}.
```

A wrong parity has zero correlation with the label, so partial overlap with `S` gives no partial credit. This exact orthogonality is what drives the SQ lower bound: one noisy query can have nontrivial correlation with only a small fraction of the possible parities, forcing essentially exhaustive search over `k`-subsets in the SQ model. Noiseless parity has a non-SQ escape route through Gaussian elimination, but noisy sparse parity is believed to require `n^{Omega(k)}` computation, and ordinary gradient training does not exploit Gaussian elimination.

Boolean Fourier analysis gives the language for the gradient signal. Any function `f : {±1}^n -> R` has a unique expansion

```text
f(x) = sum_A fhat(A) chi_A(x),   fhat(A) = E_x[f(x) chi_A(x)].
```

For a ReLU neuron, the first-layer gradient contains an indicator of a linear threshold function. At all-ones/sign initialization that indicator is `1/2 + (1/2) Majority` up to coordinate sign flips, so the relevant Fourier coefficients come from the majority spectrum. For odd coefficient order `q`,

```text
Majhat(A) = (-1)^((q-1)/2)
           * C((n-1)/2, (q-1)/2) / C(n-1, q-1)
           * 2^{-(n-1)} C(n-1, (n-1)/2),
```

the same for every `A` with `|A| = q`, and the even-order coefficients vanish. Writing `xi_q = Majhat([q])`, the adjacent coefficients needed for a `k`-sparse parity obey

```text
|xi_{k-1}| = ((n-k)/(k-1)) |xi_{k+1}|.
```

Thus the majority-function gap between the order-`k-1` and order-`k+1` coefficients is at least

```text
gamma_Maj = 0.03 (n-1)^(-(k-1)/2) = Theta(n^(-(k-1)/2))      for n >= 4k.
```

The ReLU indicator inherits the same polynomial gap, up to the harmless factor `1/2`. If an estimate of the relevant population gradient is within half of the applicable Fourier gap in `infinity` norm, the `k` largest-magnitude coordinates identify `S`. For bounded stochastic gradients, Hoeffding plus a union bound gives `O(log n / gamma^2)` samples for that one-shot recovery condition.

Low-width networks cannot fall back on fixed kernels here. Any fixed `D`-dimensional embedding with bounded feature norm needs dimension on the order of `C(n,k)` to fit all `k`-parities at margin. A two-layer MLP with width independent of `n^k` must therefore learn features by moving its first-layer weights; staying near initialization is not enough.

Two empirical facts are already on the table. In online sparse-parity training, loss and accuracy can stay flat for a long time while weight-space quantities change, and progress becomes visible only when the relevant coordinates separate from the irrelevant ones. In small finite algorithmic datasets, standard training can fit the training table long before held-out performance improves; weight decay is a strong regularizer for turning this delayed-generalization behavior into a data-efficient run, while overly strong regularization can break optimization.

## Baselines

**Online single-pass stochastic gradient descent.** At each step `t`, draw a fresh i.i.d. batch from the parity distribution and update along the minibatch loss:

```text
theta_{t+1} <- (1 - lambda_t) theta_t - eta_t grad_theta (1/B) sum_i loss(y_{t,i}, f(x_{t,i}; theta_t)).
```

Because every batch is fresh, the minibatch gradient is an unbiased estimate of the population gradient. This is the clean stochastic-approximation setting, and it avoids a train/test split inside the training distribution itself. Its limitation is sample consumption: `T` gradient steps cost `T * B` independent examples. Since the sparse-parity computation requires `n^{Omega(k)}` correlational work, this recipe pays the computational cost in fresh samples too.

**One-shot high-precision gradient identification.** If a gradient estimate at a suitable initialization is accurate enough in `infinity` norm, the Fourier gap makes feature recovery immediate: the `k` largest absolute gradient coordinates are exactly the hidden subset. This gives a sharp sufficient condition, but resolving a gap of size `gamma` costs `O(log n / gamma^2)` samples, and the clean two-layer MLP theorem uses a still larger `n^{Omega(k)}`-scale batch to support the subsequent construction. The limitation is that the guarantee spends a large fresh sample budget up front.

**Plain empirical-risk minimization on a finite sample.** A finite labeled dataset can be optimized repeatedly by the same network and optimizer. The danger is that a flexible MLP can interpolate the sample without learning the sparse parity rule. The empirical gradient is a gradient of the sample distribution, not the population distribution, and a dense interpolating solution can fit the training points while carrying little useful information off-sample. The limitation is not lack of training fit; it is lack of a bias that prefers the sparse, generalizing explanation over a dense memorizer.

## Evaluation settings

- **Task and data.** Draw a hidden `S` of size `k`. Generate training inputs uniformly from the Boolean cube and label by parity over `S`. Evaluate on fresh held-out Boolean-cube samples. Vary `n`, `k`, the number of independent training examples `m`, and the regularization strength.
- **Model.** Use the fixed two-layer MLP supplied by the harness, with a first linear layer, ReLU nonlinearity, and a scalar output layer. The width is fixed by the benchmark rather than scaled to `n^k`.
- **Optimizer and loss.** Use the fixed minibatch first-order trainer, binary parity labels in the harness encoding, and the harness optimizer family with weight decay as an exposed hyperparameter.
- **Protocol.** Hold architecture, loss, batch size, and maximum gradient-step budget fixed. Compare data-construction and weight-decay choices by train accuracy, held-out accuracy, and the gap between fitting the finite training sample and recovering the hidden rule.

## Code framework

The available substrate is a generic fixed-budget minibatch trainer. It consumes a dataset returned by `make_dataset`, reshuffles the returned examples at the start of each pass, and stops only when `max_steps` gradient updates have been taken. The empty slots are the data construction and optimizer configuration.

```python
import torch
from torch import nn


def parity_labels(x, secret):
    """Label oracle in the binary harness: sum selected bits modulo two."""
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
    """Construct the training set from the label oracle.
    TODO: choose the dataset configuration."""
    raise NotImplementedError


def get_optimizer_config():
    """Return optimizer settings for the fixed trainer.
    TODO: choose the optimizer configuration."""
    raise NotImplementedError


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
