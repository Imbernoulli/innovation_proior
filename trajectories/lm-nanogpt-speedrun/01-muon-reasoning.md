The baseline is a GPT-2 trained with AdamW, and after tuning the learning rate and swapping in rotary
position embeddings it clears 3.28 FineWeb val loss in about half an hour on the eight H100s. That's
already a big cut from the 45-minute llm.c reference, and almost all of it came from the schedule and
the optimizer's learning rate. So the obvious next lever is the optimizer. Most of the wallclock is the
optimization itself — thousands of AdamW steps grinding the loss down — and if I can make each step buy
*more* loss reduction, I reach the bar in fewer steps and the wallclock falls. The question is whether
AdamW is leaving something on the table.

Let me look at what AdamW actually does to a weight *matrix*. A linear layer is a matrix W ∈ ℝ^{m×n};
its gradient G is also m×n. Adam keeps a running estimate of the gradient's first moment m and second
moment v *per scalar entry*, and the update for each entry is roughly −lr · mᵢⱼ / (√vᵢⱼ + ε). So Adam
treats the matrix as mn independent scalars: every entry is rescaled by its own gradient magnitude and
nothing else. That's the whole trick — it's a diagonal preconditioner, normalizing each coordinate so
that flat and steep directions get comparable step sizes. It's robust precisely because it commits to
nothing about the structure of W.

But the gradient of a matrix is not a bag of scalars; it has *matrix* structure that the coordinate-wise
view throws away. Think about the singular value decomposition G = U S Vᵀ. The directions in which the
gradient is large (big singular values) and the directions in which it is small (tiny singular values)
are encoded in the singular *vectors*, not in individual entries. A coordinate-wise rescale does nothing
about a badly *conditioned* G — if the gradient's energy is concentrated in one or two singular
directions, Adam's per-entry √v normalization doesn't fix that; the update still points mostly along the
dominant singular directions, and the model effectively learns along a low-rank slice each step while the
small-singular-value directions get starved. For a transformer, where the interesting structure lives in
how the matrix maps subspaces (which query subspace reads which key subspace, etc.), that anisotropy is
exactly what I'd want the optimizer to flatten out.

So here is the move I want to make precise: instead of rescaling entries, rescale *singular values*.
Take the momentum-smoothed gradient G, compute G = U S Vᵀ, and replace S with the identity — step along
U Vᵀ. That is the "nearest orthogonal matrix" to G; it keeps the *directions* the gradient is pushing
(the singular vectors) but equalizes how hard it pushes in each of them. Every direction gets a step of
the same scale, so the small-singular-value directions are no longer starved and the dominant ones no
longer dominate. Geometrically I'm replacing "steepest descent in the Euclidean entrywise metric"
(Adam-ish) with a step that is uniform across the singular spectrum — a spectral, condition-number-blind
update. Stack standard SGD momentum in front of it so the gradient I orthogonalize is a smoothed
estimate, not a single noisy minibatch gradient.

The wall is cost. An SVD of a 768×3072 matrix, every layer, every step, on a GPU, is brutal — SVD is
sequential, doesn't love bf16, and would erase whatever per-step savings the better direction buys. If
the orthogonalization is more expensive than the AdamW step it replaces, the wallclock goes *up* even if
the loss-per-step goes down, and I've lost the race. I need the orthogonal factor U Vᵀ *without* an SVD.

What do I actually need? Given G, I want the matrix that has the same singular vectors but all singular
values equal to one — i.e. G (GᵀG)^{−1/2}, the "matrix sign" / zeroth-power / polar factor. The key
realization is that I don't need it exactly. I need it cheaply, in bf16, on a GPU, accurate *enough*
that the update direction is roughly spectrum-flat. And there's a classic tool for applying a scalar
function to the singular values of a matrix using only matrix multiplies: a polynomial iteration. If p(x)
is an odd polynomial that drives the interval (0, 1] toward 1, then applying p to the matrix — built only
from G, GGᵀ, and products — applies p to each singular value while leaving the singular vectors fixed.
Newton–Schulz is the canonical such iteration for the inverse square root / sign function. Matrix
multiplies are exactly what H100 tensor cores are fastest at, and they run fine in bf16. So I trade an
expensive exact SVD for a handful of cheap matmuls that give an *approximate* orthogonalization. That's
the unlock.

Concretely: first normalize so the top singular value is ≤ 1 (divide G by its norm), which puts every
singular value into (0, 1] where the iteration is well-behaved. Then iterate

  X ← a·X + b·(X Xᵀ X) + c·(X Xᵀ)² X

with fixed scalar coefficients (a, b, c). Each step is a degree-5 (quintic) polynomial in the singular
values: σ ↦ a·σ + b·σ³ + c·σ⁵. I want this polynomial to pull the whole interval (0, 1] up toward 1 in
as few steps as possible. The naive choice is the coefficients that make the iteration *converge* to the
exact sign function. But I don't care about exact convergence — I care about getting close in ~5 steps.
So choose (a, b, c) to maximize the *slope at zero*: the steeper p′(0), the faster the tiny singular
values (the starved directions) get yanked up toward 1 in the first couple of steps. Pushing the slope
at zero even past the point where the iteration stops converging to exactly 1 everywhere is fine — it
overshoots and leaves the singular values not at 1 but scattered around it, something like σ′ ~
Uniform(0.5, 1.5). Empirically that imperfect flattening doesn't hurt the model relative to true U Vᵀ at
all, and it lets me get away with only 5 iterations. The coefficients that do this come out to roughly
(a, b, c) = (3.4445, −4.7750, 2.0315). Five matmul-heavy steps per matrix per update — cheap on Hopper.

A couple of details fall out. The iteration as written needs G to be "tall-ish" for the X Xᵀ products to
be the smaller of the two Gram matrices; if G has more columns than rows I transpose it first and
transpose back at the end, so I always multiply the smaller Gram matrix. And after orthogonalizing, the
update U Vᵀ has all singular values ≈ 1, so its overall scale is fixed regardless of the original
gradient magnitude — I need to put a sensible scale back. I set the update so its RMS entry is order one:
multiply by max(m, n)^{1/2} (and for a fused QKV block, where three n×n matrices are stacked into a
3n×n parameter, split it and orthogonalize each n×n piece separately, then scale by n^{1/2}). Then the
actual parameter step is p ← p − lr · scale · (orthogonalized momentum).

Which parameters get this? The orthogonalization only makes sense for 2D matrices — the hidden weight
matrices of the transformer body, the attention and MLP projections. It is the wrong tool for the
embedding table and the final classifier (those are better left to a standard method) and meaningless for
1D things like biases and norm gains. So this is not a drop-in replacement for the whole optimizer: I run
it only on the transformer block matrices, and keep AdamW for the embedding/head. Two optimizers, side by
side, each stepping the parameters it's suited to. I'll name the new one after what it does — momentum,
orthogonalized by Newton–Schulz: Muon.

Why should this beat Adam on *this* race? Two reasons converge. First, the better-conditioned update
should reduce loss more per step, so I can shave iterations off the schedule and the wallclock drops.
Second — and this is the part that makes it a *speedrun* method rather than just a better optimizer — the
orthogonalization is five bf16 matmuls, which on H100 tensor cores is nearly free next to the forward and
backward pass; so I get the better direction at almost no per-step cost. The risk is that the imperfect,
overshooting Newton–Schulz makes the updates noisy enough to need a smaller learning rate or to destabilize
training, eating the gain. But the whole point of normalizing G by its norm up front and of momentum-smoothing
before orthogonalizing is to keep the iteration in the stable regime; I'll set the Muon learning rate well
below the AdamW head's and warm into it.

Here is the optimizer, run on the transformer body alongside AdamW for the embedding and head.

```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # ensure top singular value <= 1
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
    """Muon: MomentUm Orthogonalized by Newton-schulz."""
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']
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
                if g.size(0) == 3 * g.size(1):  # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5  # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)

# embedding + head on AdamW; transformer body on Muon
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0, fused=True)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
```

So the causal chain: AdamW is a diagonal preconditioner blind to a weight matrix's singular structure, so
its updates inherit the gradient's anisotropy and starve the small singular directions; orthogonalizing
the momentum-smoothed gradient (step along U Vᵀ) makes the update spectrum-flat; an exact SVD is too slow,
so a 5-step quintic Newton–Schulz with slope-at-zero-maximizing coefficients applies the orthogonalization
in bf16 matmuls that are nearly free on H100; applied to the transformer body (with AdamW kept for the
embedding/head), this should drop the loss faster per step and let me cut the iteration count, which is
the wallclock.
