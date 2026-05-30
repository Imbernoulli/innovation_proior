# Sharpness-Aware Minimization (SAM)

## Problem

For overparameterized networks, driving the training loss `L_S(w)` to its minimum is underdetermined with respect to generalization: many minima share the same low training loss but generalize very differently. The geometry around `w` — the *sharpness* of the basin — is what tracks the generalization gap. SAM trains the model to land in flat regions directly, as a drop-in replacement for the usual optimizer, without ever materializing a Hessian.

## Key idea

Instead of minimizing the loss *value* at `w`, minimize the worst-case loss in a small neighborhood of `w`, so the entire basin is low (low loss and low curvature):

```
min_w  L_S^SAM(w) + λ‖w‖²,        L_S^SAM(w) ≜ max_{‖ε‖_p ≤ ρ} L_S(w + ε).
```

This objective is motivated by a PAC-Bayes bound: with high probability,
`L_D(w) ≤ max_{‖ε‖₂≤ρ} L_S(w+ε) + h(‖w‖²/ρ²)` with `h` strictly increasing, so the max-in-a-ball term plus a weight-norm penalty is a genuine upper bound on the population loss. The bracketed quantity `max_{‖ε‖≤ρ} L_S(w+ε) − L_S(w)` is the sharpness; the `h(·)` term is replaced in practice by standard L2 weight decay `λ‖w‖²`.

## Deriving the update

**Inner maximization (linearize).** First-order Taylor in `ε` around `0`:
`L_S(w+ε) ≈ L_S(w) + ε^T ∇_w L_S(w)`. Dropping the constant, the inner maximizer solves `argmax_{‖ε‖_p≤ρ} ε^T g` with `g = ∇_w L_S(w)`. This is a dual-norm problem; by Hölder (`1/p + 1/q = 1`):

```
ε̂(w) = ρ · sign(g) · |g|^{q-1} / (‖g‖_q^q)^{1/p}.
```

- `p = 2` (`q = 2`, default):  `ε̂ = ρ · g / ‖g‖₂`   (rescale the gradient to norm ρ).
- `p = ∞` (`q = 1`):  `ε̂ = ρ · sign(g)`.

`p = 2` is the default — it matches the bound's derivation and empirically beats both `p = ∞` and a random perturbation of equal norm.

**Outer gradient (drop second order).** Differentiating `L_S^SAM(w) ≈ L_S(w + ε̂(w))`:

```
∇_w L_S^SAM(w) ≈ ∇_w L_S(w)|_{w+ε̂(w)} + (dε̂/dw)^T ∇_w L_S(w)|_{w+ε̂(w)}.
```

The second term involves `dε̂/dw`, hence the Hessian (only via Hessian-vector products, so it is tractable). It is dropped — both to halve the cost and because the resulting first-order update generalizes at least as well as keeping the term. Final gradient:

```
∇_w L_S^SAM(w) ≈ ∇_w L_S(w)|_{w + ε̂(w)}.
```

i.e. the ordinary gradient, evaluated at the *ascended* point `w + ε̂(w)`, used to update `w`.

## Algorithm (per step, base optimizer = SGD)

```
sample batch B
g      = ∇_w L_B(w)                       # 1st forward-backward at w
ε̂      = ρ · g / (‖g‖₂ + 1e-12)
w'     = w + ε̂                            # ascend to the local worst case
g_SAM  = ∇_w L_B(w)|_{w'}                  # 2nd forward-backward at w'
w      ← w − η · g_SAM                     # descend at w (via base optimizer)
```

Two gradient evaluations per step (≈2× cost; compare against a baseline run for 2× epochs). One hyperparameter `ρ` (default `0.05`; grid `{0.01,0.02,0.05,0.1,0.2,0.5}`).

**m-sharpness.** In practice `ε̂` is computed per-batch, or per-accelerator-shard of size `m`, and the resulting SAM gradients are averaged (perturbations are *not* synced across shards). This corresponds to summing independent `ε`-maximizations over disjoint `m`-subsets. Smaller `m` tends to generalize better and correlate more strongly with the generalization gap — a free benefit that aligns with data-parallel scaling.

## Code

```python
import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization, as an optimizer wrapper around any base optimizer."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # ascend to w + e_hat, where e_hat = rho * g / ||g||_2
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)                       # climb to the local maximum "w + e(w)"
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # restore w, then take the real (sharpness-aware) descent step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]   # back to "w" from "w + e(w)"
        self.base_optimizer.step()                # uses the gradient computed at w + e(w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires a closure that does a full forward-backward pass"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()                                 # recompute the gradient at w + e(w)
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
```

```python
# training loop
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=0.05,
                lr=0.1, momentum=0.9, weight_decay=5e-4)

for x, y in loader:
    # 1st forward-backward at w: gradient used to build e_hat
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # 2nd forward-backward at w + e_hat: the sharpness-aware gradient
    loss_fn(model(x), y).backward()
    optimizer.second_step(zero_grad=True)
```

With batch normalization, run the second forward pass without updating the running statistics (e.g. temporarily zero the BN momentum) so the perturbed pass does not pollute the BN estimates.
