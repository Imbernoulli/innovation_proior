# Deep Matrix Factorization, distilled

Deep Matrix Factorization completes a low-rank matrix by fitting observed entries with a full-dimensional
depth-`N` linear factorization

```text
W = W_N W_{N-1} ... W_1
```

using near-zero initialization and small-step optimization, with no explicit rank cap or regularizer.
Depth changes the trajectory: under balanced gradient flow,

```text
dot sigma_r(t) =
  -N (sigma_r(t)^2)^(1 - 1/N) <grad ell(W(t)), u_r(t) v_r(t)^T>.
```

The multiplier `N sigma_r^(2-2/N)` accelerates already-large singular modes and throttles small ones. With
near-zero initialization, only data-aligned modes switch on, while weaker modes stay small; depth `N >= 3`
turns the two-mode toy dynamics from a polynomial gap into finite saturation of the weaker mode.

The norm-minimization story is deliberately not the answer. In commuting PSD matrix sensing, the
Gunasekar-style nuclear-norm proof extends to every depth `N >= 3`, and the resulting nuclear-norm point can
fail to be a local minimizer of any Schatten-`p` quasi-norm for `0 < p < 1`. The bias is therefore a
trajectory effect, not a single Schatten penalty.

## Defaults

- `depth = 3`: cheapest depth with the saturation-style low-rank effect; depth 4 was empirically similar.
- `init_scale = 1e-3`: puts the end-to-end product near zero so singular modes start throttled.
- Per-factor Gaussian std: `init_scale^(1/depth) * n^(-1/2)`, matching the reference Gaussian branch.
- Full hidden dimension: no explicit rank cap.
- Train observed-entry MSE to about `1e-6` or until the iteration budget.
- Canonical optimizer path: `GroupRMSprop` with `eps=1e-4` and matrix-completion config `lr=1e-3`; SGD and
  Adam are reference-code options, but Adam is not the default recipe.

## Reference-faithful code

```python
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class GroupRMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-6):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        defaults = dict(lr=lr, alpha=alpha, eps=eps, adjusted_lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state
            if len(state) == 0:
                state["step"] = 0
                state["square_avg"] = torch.tensor(0.0)

            square_avg = state["square_avg"]
            alpha = group["alpha"]
            square_avg.mul_(alpha)
            state["step"] += 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("GroupRMSprop does not support sparse gradients")
                square_avg.add_((1 - alpha) * grad.pow(2).sum().cpu().float())

            avg = square_avg.div(1 - alpha ** state["step"]).sqrt_().add_(group["eps"])
            lr = group["lr"] / avg
            group["adjusted_lr"] = lr

            for p in group["params"]:
                if p.grad is not None:
                    p.data.add_(-lr.to(p.grad.data.device) * p.grad.data)

        return loss


class DeepMatrixFactorization:
    def __init__(self, depth=3, init_scale=1e-3, lr=1e-3, train_thres=1e-6):
        if depth < 2:
            raise ValueError("depth must be at least 2")
        self.depth = int(depth)
        self.init_scale = float(init_scale)
        self.lr = float(lr)
        self.train_thres = float(train_thres)

    def _build(self, n, device):
        layers = [nn.Linear(n, n, bias=False) for _ in range(self.depth)]
        scale = (self.init_scale ** (1.0 / self.depth)) * (n ** -0.5)
        for layer in layers:
            nn.init.normal_(layer.weight, std=scale)
        return nn.Sequential(*layers).to(device)

    @staticmethod
    def _e2e(model):
        weight = None
        for layer in model.children():
            weight = layer.weight.t() if weight is None else layer(weight)
        return weight

    def recover(self, observed_values, observed_mask, n, rank_hint, device, max_iters, log_iters):
        model = self._build(n, device)
        optimizer = GroupRMSprop(model.parameters(), self.lr, eps=1e-4)
        mask = observed_mask.to(device)
        target = observed_values.to(device)
        denom = max(int(mask.sum().item()), 1)

        for it in range(max_iters):
            e2e = self._e2e(model)
            loss = ((e2e - target) * mask).pow(2).sum() / denom
            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                if loss.item() <= self.train_thres:
                    break

            optimizer.step()

        with torch.no_grad():
            return self._e2e(model).detach().cpu()


def build_strategy():
    return DeepMatrixFactorization(depth=3, init_scale=1e-3, lr=1e-3, train_thres=1e-6)
```
