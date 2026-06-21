# mixup: Beyond Empirical Risk Minimization

Given supervised examples `(x_i, y_i)`, train on virtual pairs formed by the same convex coefficient in input and target space:

```text
lambda ~ Beta(alpha, alpha), alpha > 0
i, j ~ uniform training indices
x_tilde = lambda x_i + (1 - lambda) x_j
y_tilde = lambda y_i + (1 - lambda) y_j
minimize ell(f(x_tilde), y_tilde)
```

Equivalently:

```text
R_mix(f) = E_{i,j,lambda} ell(f(lambda x_i + (1 - lambda)x_j),
                              lambda y_i + (1 - lambda)y_j)
```

As `alpha -> 0`, `Beta(alpha, alpha)` concentrates at the endpoints, so the sampled virtual examples approach ordinary empirical examples. In implementation, the `alpha <= 0` case is handled by setting `lambda = 1`, which directly recovers ordinary minibatch training.

The method's prior is:

```text
linear movement in input space should induce linear movement in target space
```

Averaging inputs alone creates an off-sample location without the right supervision. Softening labels alone creates an input-independent confidence penalty. The method is the coupling: the same `lambda` determines both the virtual input and the virtual target.

For classification with cross-entropy, the dense soft target does not require a custom loss:

```text
CE(p, lambda e_a + (1 - lambda)e_b)
= lambda CE(p, e_a) + (1 - lambda)CE(p, e_b)
```

That gives the canonical minibatch implementation shape:

```python
import numpy as np
import torch


def mixup_data(x, y, alpha=1.0):
    """Return mixed inputs, paired targets, and the scalar interpolation weight."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Cross-entropy against the two-point soft target."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
```

The proof sketch in the source studies the averaged predictor:

```text
tilde f(x) = E_{x'',lambda} hat f(lambda x + (1 - lambda)x'')
```

Assume `hat f` has zero error on the virtual examples and measure complexity by the empirical Lipschitz constant over real training inputs:

```text
hat Lip(g) = sup_{x,x' in D} ||g(x') - g(x)|| / ||x' - x||
```

Then:

```text
hat f(lambda x' + (1 - lambda)x'') - hat f(lambda x + (1 - lambda)x'')
= [lambda f(x') + (1 - lambda)f(x'')]
  - [lambda f(x) + (1 - lambda)f(x'')]
= lambda(f(x') - f(x))
```

Therefore:

```text
hat Lip(tilde f) <= E[lambda] hat Lip(f)
```

For `lambda ~ Beta(alpha, alpha)` with `alpha > 0`, `E[lambda] = 1/2`. The bound's important content is the cancellation enabled by paired target mixing; the changing regularization strength of `alpha` comes from where the sampled virtual points concentrate, not from an alpha-varying mean in this symmetric-beta proof.

Operational checks:

- Same-class pairs are handled by the same formula; if `y_i = y_j`, the target stays that class.
- Cross-class pairs receive a coefficient-dependent soft target, avoiding midpoint label discontinuities.
- If `i = j`, the virtual example is exactly the original example for any `lambda`.
- If `lambda` is near `0` or `1`, the virtual example is near one endpoint and the target is near that endpoint's label.
- The canonical code uses one scalar `lambda` per minibatch and a random permutation of the minibatch, not per-example lambdas or `max(lambda, 1 - lambda)`.
