**Problem (from step 6).** Back at the 94% bar (airbench94: 3.83 s / 3.29 s compiled), the one untouched
lever is the *update rule*. The conv weights — most of the parameters, and where the features are learned —
are still updated by plain Nesterov SGD, which steps in the raw gradient direction. A conv weight reshaped
to 2-D is a matrix whose gradient has a spectrum of wildly unequal singular values, so SGD over-updates the
few dominant singular directions and starves the rest. In a ~10-epoch run the starved directions never catch
up.

**Key idea.** The **Muon** optimizer (MomentUm Orthogonalized by Newton-schulz): apply momentum, then
*orthogonalize* the momentum buffer — replace gradient matrix G = U S Vᵀ with U Vᵀ, flattening all singular
values to ≈1 so every direction of the weight matrix gets equal step length (steepest descent in the
spectral norm, the natural geometry for a matrix parameter). Compute the orthogonalization SVD-free via a
3-step **Newton–Schulz iteration** of all-matmuls with coefficients (3.4445, −4.7750, 2.0315) chosen to
maximize the slope at zero (so small singular values shoot up fast). Apply Muon only to the 4-D conv filters;
keep scalars/biases/head on plain SGD. Renormalize each weight to fixed norm before each scale-free step.

**Why it works.** Orthogonalized updates equalize per-direction learning across each weight matrix, so the
low-singular-value directions a short SGD run leaves starved all train at once — clearing 94% in fewer steps
and dropping the record below 3.29 s. The Newton–Schulz iteration needn't converge cleanly (it yields U S' Vᵀ
with S' ≈ Uniform(0.5, 1.5), which empirically doesn't hurt) and is pure matmul, so its per-step cost is
repaid by needing fewer steps. This is the fastest documented 94%-on-CIFAR-10 record.

**Change / code.** The Newton–Schulz orthogonalization and the Muon step.

```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)            # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def step(self):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None: continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                p.data.mul_(len(p.data)**0.5 / p.data.norm())                          # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)  # whiten the update
                p.data.add_(update, alpha=-lr)                                         # take a step

# conv filters -> Muon; biases/head -> SGD
filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
```

**Target.** Clear the 94% bar faster than the standing record (airbench94, 3.83 s / 3.29 s compiled), making
this the fastest known training to 94% on CIFAR-10.
