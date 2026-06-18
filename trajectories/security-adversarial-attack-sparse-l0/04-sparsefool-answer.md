**Problem (from step 3).** OnePixel's black-box population search escaped JSMA's local optima (mean ASR
≈ 0.153) but was *query-starved* — six DE generations over a 120-dim encoding harvest only the easy flips.
Spend the budget far more efficiently: go back to the gradient, but through a reconsiderable,
boundary-aware construction, not a greedy saliency walk.

**Key idea (SparseFool).** Relax `L0` (NP-hard) to its convex `L1` surrogate. `L1`'s dual is `L_inf`, so
projecting onto a hyperplane spends all mass on one coordinate — sparse for free. Linearize the *decision
boundary* (low local curvature near data) at an `L2`-DeepFool boundary point `x_B` with oriented normal
`w = grad f_adv(x_B) - grad f_true(x_B)`, then solve `min ||r||_1` s.t. `w^T(x+r-x_B)=0`, `l<=x+r<=u` by a
coordinate-greedy projection that *retires* any coordinate saturating against the box. Relinearize at each
new iterate until the label flips — the reconsideration JSMA lacked.

**Why it beats the prior rungs.** Directed boundary-following uses the gradient far more efficiently than
undirected DE sampling (OnePixel's weakness), and relinearizing every step avoids greedy commitment
(JSMA's weakness). The box lives *inside* the optimization, so validity never collapses the way an end-clip
on `L1`-DeepFool does.

**Scaffold edit / hyperparameters.** Thin wrapper around `torchattacks.SparseFool`, `steps=20` (outer
relinearizations — far more *effective* than OnePixel's 6 DE generations per query), `overshoot=0.02`
(final cross-the-boundary nudge), `lam=3.0` (the one real knob: aims the solver target *past* the boundary
to absorb curvature; the aggressive CIFAR-10 setting, chosen because robust-model boundaries are more
curved). `pixels`/`device`/`n_classes` unused; harness validates `L0 ≤ 24`.

**What to watch.** Mean ASR beats 0.153 but modestly (high-teens), and may *trail* on the least-linear
model — SparseFool's local-linear boundary model is brittle on adversarially-trained surfaces. That points
the final rung at a method built natively for the discrete `L0` set.

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torchattacks

    _ = (pixels, device, n_classes)
    model.eval()
    attack = torchattacks.SparseFool(
        model,
        steps=20,
        lam=3.0,
        overshoot=0.02,
    )
    return attack(images, labels)
```
