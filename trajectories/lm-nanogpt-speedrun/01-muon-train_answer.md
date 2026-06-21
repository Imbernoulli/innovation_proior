The AdamW + rotary GPT-2 recipe clears the 3.28 FineWeb val-loss bar in about 31 minutes on the eight H100s, and when I profile a run almost all of that wallclock is the optimization itself — thousands of steps grinding the loss down. So the lever on wallclock is the optimizer: if each step buys *more* loss reduction, I reach the bar in fewer steps and the run gets shorter. The question is whether AdamW is leaving something on the table, and looking at what it does to a weight *matrix* says it is. A linear layer is a matrix $W \in \mathbb{R}^{m\times n}$ with gradient $G$ of the same shape, and Adam keeps first/second moment estimates *per scalar entry*, updating each by roughly $-\eta\, m_{ij}/(\sqrt{v_{ij}}+\epsilon)$. It treats the matrix as $mn$ independent scalars — a diagonal preconditioner that rescales each coordinate by its own gradient magnitude and is otherwise blind to the matrix structure. That blindness is the failure: the gradient of a matrix has structure encoded in its singular value decomposition $G = U S V^\top$, and a per-entry rescale does nothing about a badly *conditioned* $G$. If the gradient's energy concentrates in a few singular directions, the update still points mostly along those dominant directions and the small-singular-value directions get starved — and for a transformer, where the interesting structure lives in how a matrix maps subspaces, that anisotropy is exactly what I want the optimizer to flatten.

I propose **Muon** — *MomentUm Orthogonalized by Newton-schulz*. The idea is to stop rescaling entries and instead rescale *singular values*: take the momentum-smoothed gradient $G$, and instead of stepping along it, step along its nearest orthogonal matrix $U V^\top$ (where $G = U S V^\top$). This keeps the *directions* the gradient is pushing — the singular vectors — but replaces the singular values $S$ with the identity, so every singular direction gets a step of the same scale. The dominant directions no longer dominate and the starved directions are no longer starved; geometrically I have replaced Euclidean entrywise steepest descent with a spectrum-flat, condition-number-blind update. Standard SGD momentum sits in front of the orthogonalization so the matrix I orthogonalize is a smoothed estimate, not a single noisy minibatch gradient.

What makes it a *speedrun* method rather than just a better optimizer is that the orthogonalization is cheap. The obvious way to get $U V^\top$ is an SVD per matrix per step, but SVD is sequential, dislikes bf16, and on GPU would erase every per-step saving the better direction buys — if orthogonalizing costs more than the AdamW step it replaces, the wallclock goes *up*. So I compute the orthogonal factor approximately, without an SVD. What I want is the matrix with the same singular vectors but all singular values equal to one, the polar factor $G(G^\top G)^{-1/2}$, and I only need it *roughly*, cheaply, in bf16. An odd polynomial $p(x)$ that drives the interval $(0,1]$ toward $1$, applied to the matrix through only $G$, $GG^\top$ and their products, applies $p$ to each singular value while leaving the singular vectors fixed — and matrix multiplies are exactly what H100 tensor cores are fastest at. This is a Newton–Schulz iteration. First I normalize so the top singular value is $\le 1$ by dividing $G$ by its norm, putting every singular value into $(0,1]$ where the iteration is well-behaved; then I iterate

$$X \leftarrow a\,X + b\,(XX^\top X) + c\,(XX^\top)^2 X,$$

a quintic whose action on each singular value is $\sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5$. The load-bearing choice is the coefficients. I do not pick the values that make the iteration *converge* to the exact sign function — I do not care about exact convergence, I care about getting close in about five steps. So I choose $(a,b,c)$ to *maximize the slope at zero*: the steeper $p'(0)$, the faster the tiny singular values get yanked toward $1$ in the first couple of steps. Pushing the slope past the point where the iteration still converges everywhere to exactly $1$ is fine — it overshoots and leaves $\sigma' \sim \text{Uniform}(0.5, 1.5)$ rather than exactly $1$, and empirically that imperfect flattening works as well as true $U V^\top$ while letting me get away with only five iterations. The coefficients come out to roughly $(a,b,c) = (3.4445, -4.7750, 2.0315)$.

Two details fall out. The iteration wants $G$ tall-ish so the $XX^\top$ products are the smaller Gram matrix; if $G$ has more columns than rows I transpose it first and transpose back at the end. And after orthogonalizing, the update has all singular values $\approx 1$, so its overall scale is fixed regardless of the original gradient magnitude — I restore a sensible scale by multiplying by $\max(m,n)^{1/2}$ so the update's RMS entry is order one (for a fused QKV block, three $n\times n$ matrices stacked into a $3n\times n$ parameter, I split it and orthogonalize each $n\times n$ piece, scaling by $n^{1/2}$). The orthogonalization only makes sense for 2D matrices — the transformer body's attention and MLP projections — so Muon is *not* a drop-in for the whole optimizer; it is the wrong tool for the embedding table and the final classifier and meaningless for 1D biases and norm gains. I run it only on the transformer block matrices and keep AdamW for the embedding/head, two optimizers side by side. Five bf16 matmuls per matrix are nearly free next to the forward and backward pass on H100, so I get the better-conditioned direction at almost no per-step cost while reducing loss more per step — which lets me shorten the schedule, and the schedule is the wallclock. The Muon learning rate is set well below the AdamW head's and warmed in, which together with the norm-normalization and momentum-smoothing keeps the iteration in its stable regime.

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
