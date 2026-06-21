# Sharpness-Aware Minimization

Use the neighborhood loss, not only the point loss:

```text
min_w  L_S^SAM(w) + lambda ||w||_2^2

L_S^SAM(w) = max_{||epsilon||_p <= rho} L_S(w + epsilon)
```

The inner maximization is approximated by first-order expansion at `w`. With `g = grad_w L_S(w)` and `1/p + 1/q = 1`:

```text
epsilon_hat =
  rho * sign(g) * |g|^(q - 1) / (||g||_q^q)^(1/p)
```

For the default `p = 2`:

```text
epsilon_hat = rho * g / (||g||_2 + eps)
```

The training step is:

```text
1. Compute g = grad_w L_B(w).
2. Ascend to w_adv = w + epsilon_hat.
3. Compute g_sam = grad_w L_B(w_adv).
4. Restore w and update it with the base optimizer using g_sam.
```

This costs two gradient evaluations per step. The distinctive part is the adversarial weight perturbation: the update descends from the worst nearby loss direction, so it suppresses sharp local loss increases. Weight decay remains only as the norm-control term suggested by the bound; random noise is a weaker foil because it samples directions instead of attacking the locally steep one.

## Minimal SAM Optimizer Wrapper

```python
class SAM:
    def __init__(self, params, base_optimizer, rho=0.05):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.state = {}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for p in self.params:
            if p.grad is None:
                continue
            e_w = self.rho * p.grad / (grad_norm + 1e-12)
            self.state[p] = e_w
            p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.params:
            if p.grad is None:
                continue
            p.sub_(self.state[p])
        self.base_optimizer.step()
        self.state.clear()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        return torch.norm(
            torch.stack([p.grad.norm(p=2) for p in self.params if p.grad is not None])
        )

    def zero_grad(self):
        self.base_optimizer.zero_grad()
```
