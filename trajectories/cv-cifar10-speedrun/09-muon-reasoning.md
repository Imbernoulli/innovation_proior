Back to the 94% bar, which is where the headline record lives, and where I left it at airbench94: 3.83
seconds, 3.29 compiled. Every algorithmic lever I pulled to get there — whitening, Dirac init, the 64×
bias lr, Lookahead, multi-crop TTA, alternating flip — has been about *initialization*, *data*, or
*inference*. The one thing I have left almost untouched is the *update rule itself*. The conv weights,
which are the vast bulk of the parameters and the part that actually learns the features, are still being
updated by plain Nesterov SGD: take the (momentum-smoothed) gradient and step in its direction. I want to
ask whether SGD is leaving speed on the table in *how* it turns a gradient into a step for these weight
*matrices*.

Here's the geometric issue. A convolution's weight, reshaped to 2-D, is a matrix mapping input features
to output features. Its gradient is also a matrix, and that gradient matrix has a *spectrum* — singular
values that can be wildly unequal. SGD steps in the raw gradient direction, which means the update is
dominated by the gradient's top singular directions: the few directions where the gradient happens to be
large get most of the step, and the many directions where the gradient is small barely move. So early
training over-updates a handful of dominant directions in each weight matrix and starves the rest. In a
long run the starved directions eventually get their turn; in a ~10-epoch run they may never catch up.
What I'd prefer is an update that treats all the directions of the weight matrix *evenly* — that moves
the same distance along every direction the gradient points in, regardless of how large or small that
direction's gradient magnitude is.

The mathematical object that does this is the *orthogonalization* of the gradient. If the (momentum-
smoothed) gradient matrix G has SVD G = U S Vᵀ, then the matrix U Vᵀ — G with all its singular values
flattened to one — points in the same "directions" as G but gives every direction equal step length.
Updating with U Vᵀ instead of G is a steepest-descent step under the spectral (operator) norm rather than
the Euclidean norm; it's the natural geometry for a *matrix* parameter, and it equalizes the per-direction
learning the way the 64× bias lr equalized learning between bias and weight parameters — same theme,
different axis. The problem is that computing an SVD of every conv weight every step is far too slow for a
2.59-second budget; the whole record would drown in eigendecompositions.

So I need an SVD-free way to compute (approximately) U Vᵀ. The trick is a *Newton–Schulz iteration*: a
fixed polynomial in G applied a few times that drives the singular values toward 1 while leaving the
singular vectors alone. Starting from G normalized so its top singular value is ≤ 1, iterating
`X ← a·X + (b·(XXᵀ) + c·(XXᵀ)²)·X` with carefully chosen coefficients (a, b, c) = (3.4445, −4.7750,
2.0315) pushes the spectrum toward 1 in just a handful of steps — and it's all matmuls, which the GPU
eats for breakfast. I don't even need it to converge cleanly to exactly U Vᵀ: I can pick the
coefficients to maximize the slope at zero (so small singular values shoot up fast) even past the point
where the iteration stops converging to exactly 1 everywhere. That leaves something like U S' Vᵀ with
S' randomly spread in roughly (0.5, 1.5) — not a clean orthogonalization, but in practice it doesn't
hurt at all, and three iterations suffice. This is the **Muon** optimizer (MomentUm Orthogonalized by
Newton-schulz): momentum on the gradient, then orthogonalize the momentum buffer via Newton–Schulz,
then step.

```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)            # ensure top singular value <= 1
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X
```

I apply Muon only to the 4-D conv filters (reshaped to 2-D) — the matrices where the spectral argument
bites — and keep the cheap, well-behaved scalars (the whitening bias, the norm biases, the linear head)
on plain SGD. One more piece pairs naturally with orthogonalized updates: since the update direction is
now scale-free (singular values ≈ 1), the *magnitude* of each weight matters independently, so before each
orthogonalized step I renormalize the weight to a fixed norm — `p.mul_(len(p)**0.5 / p.norm())` — keeping
every filter matrix at a controlled scale so the equal-length updates land consistently. Momentum buffer,
Newton–Schulz orthogonalization, weight normalization, step:

```python
buf.mul_(momentum).add_(g)
g = g.add(buf, alpha=momentum) if nesterov else buf
p.data.mul_(len(p.data)**0.5 / p.data.norm())                 # normalize the weight
update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)  # whiten the update
p.data.add_(update, alpha=-lr)                                 # take a step
```

This is the closing move, so let me set the bar it has to clear and why I believe it can. The standing
94% record at the top of this ladder is airbench94 at 3.83 seconds (3.29 compiled). The bet is that
replacing SGD with Muon on the conv weights equalizes the per-direction learning of every weight matrix —
so the starved low-singular-value directions, which a 10-epoch SGD run never finishes training, all learn
at once — and that this lets the same architecture clear 94% in *fewer steps still*, dropping the record
below 3.29 seconds. The two hedges are real: the Newton–Schulz iteration is approximate (it produces
US'Vᵀ, not UVᵀ), and it adds a few matmuls per weight per step, so the per-step cost goes up and must be
repaid by needing fewer steps. The wager of this whole ladder has been that the right *structural* insight
beats brute force, and the update geometry is the last structure left to fix; if orthogonalizing the
matrix updates buys back more steps than the iteration costs, Muon is the fastest way known to train a net
to 94% on CIFAR-10, and that is where the ladder ends.
