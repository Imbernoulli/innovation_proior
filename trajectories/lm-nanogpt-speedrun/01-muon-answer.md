**Problem (from the baseline).** The AdamW + rotary GPT-2 recipe reaches 3.28 FineWeb val loss in ~31
minutes, and almost all the wallclock is the optimization itself. AdamW is a coordinate-wise diagonal
preconditioner: it rescales each scalar entry of a weight matrix by its own running gradient statistics
and is blind to the matrix's singular structure, so its updates inherit the gradient's anisotropy and
underweight the small-singular-value directions. Making each step buy more loss is the lever on wallclock.

**Key idea (Muon).** Orthogonalize the momentum-smoothed gradient before stepping: replace the
SGD-momentum update G with the nearest orthogonal matrix U Vᵀ (where G = U S Vᵀ), keeping the gradient's
singular *vectors* but equalizing its singular *values*. This is a spectral, condition-number-blind step
— every singular direction gets a step of the same scale. An exact SVD per matrix per step is too slow on
GPU, so compute the orthogonal factor approximately with a 5-step quintic Newton–Schulz iteration built
only from matrix multiplies (cheap in bf16 on Hopper). The coefficients (3.4445, −4.7750, 2.0315) are
chosen to maximize the polynomial's slope at zero — pulling tiny singular values toward 1 fast — even
past exact convergence; the result is roughly U S′ Vᵀ with S′ ≈ Uniform(0.5, 1.5), which works as well as
true U Vᵀ. Run it only on the 2D transformer-body matrices; keep AdamW for the embedding and head.

**Why it works.** Five bf16 matmuls per matrix are nearly free next to the forward/backward pass on H100,
so the better-conditioned update direction comes at almost no per-step cost while reducing loss more per
step — which lets the schedule be shortened, dropping the wallclock to the 3.28 bar. Normalizing G by its
norm and momentum-smoothing before orthogonalizing keep the iteration in its stable regime; the Muon LR is
set well below the AdamW head's and warmed in.

**Change / code.** New `Muon` optimizer on `transformer.h`, AdamW retained for `lm_head`; tall/wide
transpose so the smaller Gram matrix is multiplied; per-update scale `max(m,n)^{1/2}` so the orthogonalized
update has RMS entry ≈ 1; fused-QKV blocks split and orthogonalized per n×n piece.

```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """Newton-Schulz quintic iteration: orthogonalize G (zeroth power) in bf16 via matmuls.
    Coefficients maximize slope at zero; result ~ U S' V^T with S' ~ Uniform(0.5,1.5)."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz. For 2D params only."""
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps))

    def step(self):
        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):                      # fused QKV: orthogonalize each n×n block
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5          # RMS entry ≈ 1
                p.data.add_(g, alpha=-lr * scale)

# transformer body on Muon; embedding/head on AdamW
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0, fused=True)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
```
