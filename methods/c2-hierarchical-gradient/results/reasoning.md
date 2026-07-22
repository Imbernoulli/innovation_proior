The coarse annealing gave me a `20`-piece profile at `0.8848`, just under the published `0.88922`, and the
feedback was clear about why it stops there: `20` pieces is too coarse a grid to render a sufficiently
flat-topped autoconvolution, and the cap on how flat that cap can get is the cap on `R`. The optimized
shape itself looks right — a tall spike plus a tapering shoulder, with several heights pinned at zero — so I
do not want to throw it away and search a long vector from scratch. What I want is to *keep this shape* and
give it more resolution, so the autoconvolution can develop the fine structure that a coarse vector simply
cannot represent.

The obvious way to add resolution without disturbing the shape is to split each piece into `k` equal copies.
A height vector `v` of length `20` becomes `repeat(v, k)` of length `20k`, and as a function on the line it
is the *same graph* — same steps, same heights, just relabelled at finer granularity. If that is right then
`R` should not move at all under the upscale. I should check this rather than assume it, because the scoring
goes through a self-convolution and a max, and it is easy to imagine an off-by-one in the node indexing
making the upscaled `R` drift. So I take a little vector `v = [0.2, 1.0, 0.7, 0.3, 0.0]`, compute `R(v)`, and
compare against `R(repeat(v, k))` for `k = 2, 3, 5`:

```
R(v)          = 0.7064665796723212
R(repeat x2)  = 0.7064665796723211   diff = -1.1e-16
R(repeat x3)  = 0.7064665796723211   diff = -1.1e-16
R(repeat x5)  = 0.7064665796723211   diff = -1.1e-16
```

The differences are at the last bit of the double, i.e. zero up to rounding, and sweeping `200` random
vectors at random `k` the worst gap is `6.7e-16`. So the upscale is ratio-preserving to machine precision —
it costs nothing and risks nothing, it just hands the optimizer a finer canvas. That settles the lift step.

The harder question is what to do *after* the lift. At the upscaled point every block of `k` copies is flat,
so the function sits at a special, symmetric configuration, and I need an optimizer that will break that
symmetry and let neighbouring pieces within a block differ — that is where the new degrees of freedom, and
any new gains, must live. At `N` in the hundreds a coordinate-wise annealing that perturbs one height at a
time is hopeless: tens of thousands of acceptance tests to make one coherent change across a shoulder. I want
to move *all* the heights together along a good direction, which means I need a gradient of `R` with respect
to the height vector.

The obstruction is the `max` in the denominator: `||f*f||_inf = max_j L_j` is not differentiable at the peak.
The standard fix is to replace the hard max by the log-sum-exp softmax `B(β) = m + log Σ_j exp(β(L_j−m))/β`
with a sharpness `β`, which is smooth everywhere, and then *anneal* `β` upward over the run — soft early, where
the surrogate is smooth and the gradient points broadly toward better shapes, sharp late, where the surrogate
should be a faithful stand-in for the true `max`. "Should be" is doing work there, so I check how fast `B(β)`
closes on `max_j L_j` for an actual `L`. On a random `20`-piece vector whose autoconvolution peaks at
`m = 7.107858`:

```
beta=   5   B=7.239784   (B-max)/max = 1.9e-02
beta=  40   B=7.107864   (B-max)/max = 8.3e-07
beta= 400   B=7.107858   (B-max)/max = 0
beta=6000   B=7.107858   (B-max)/max = 0
```

By `β = 400` the surrogate denominator already agrees with the true `max` to all printed digits, and the
coarse seed itself was annealed out to `β = 6000`. So sharpening `β` into the thousands does make the
surrogate optimum the true-ratio optimum; the soft-to-sharp schedule is buying genuine smoothing early and
fidelity late, not just relabelling the objective.

Now the gradient itself. The chain rule has two hops: `dR/dL_j` from the closed-form norms, then `dL_j/dv`
from the self-convolution. Because each `L_j = (v*v)_{j−1}`, the derivative of any scalar with respect to `v`
is a convolution of the node-gradient with `v` reflected — `g = 2·(dL ⋆ v)` with appropriate truncation. That
is short to write but exactly the kind of expression where a reflection index or a factor of two goes wrong
silently, so I do not trust it until it matches a finite difference. With `β = 40` and a random `8`-vector:

```
analytic g : [-0.001574 -0.079097  0.079205  0.156721 -0.082493  0.170985 -0.133605 -0.061019]
finite-diff: [-0.001574 -0.079097  0.079205  0.156721 -0.082493  0.170985 -0.133605 -0.061019]
max abs err: 1.7e-10
```

Agreement to `1e-10`, which is finite-difference accuracy — the analytic gradient is correct, including the
factor `2` and the reflected-convolution truncation. Everything is `O(N log N)` with FFTs, so a few thousand
gradient steps at `N = 500` is seconds.

For the optimizer on top of this gradient I reach for Adam rather than plain ascent, and the reason is in the
data, not in taste: the optimized coarse seed has a spike about `19×` the smallest nonzero shoulder height
(I measured the ratio on the actual `2024`-seed coarse vector), so the heights span more than an order of
magnitude. A single fixed step either crawls on the spike or blows up the thin shoulder; Adam's per-coordinate
adaptive scaling normalizes each height's step by its own recent gradient magnitude, so spike and shoulder
advance on comparable terms. After every step I clip to non-negative to stay legal, and I track the best
*true* `R` ever seen rather than the surrogate, since the two differ slightly and I want to return the
genuinely best vector.

That leaves the symmetry of the upscaled point. My first instinct was that the upscaled vector is a degenerate
plateau: within a block all `k` copies are equal, so I assumed any antisymmetric within-block direction would
have zero gradient, the configuration would be a fixed point of the refinement, and Adam dropped there would
sit still until something perturbed it. The natural remedy is a small multiplicative kick right after each
upscale to break the block symmetry. Before building the kick in as load-bearing, I tested the premise: take
a coarse vector, upscale `×5`, and look at the gradient reshaped into `(block, within-block)` form. If the
plateau story is right the gradient should be constant across the `5` entries of each block. It is not —
each block's `5` gradient entries spread by `0.01`–`0.03`, comparable to the block-to-block variation, and the
within-block (mean-subtracted) component of the first gradient has norm `0.017` against `0.032` for the
block-mean component. So the within-block gradient is plainly nonzero. That makes sense in hindsight: the
self-convolution couples copies across blocks, so even though the *heights* within a block are equal, the
*gradient* at each of those equal heights is different because each sits at a different position in the
convolution. The flat block is not a critical point of `R`.

So the "Adam sits still" story is wrong, and I should not let the kick rest on it. I ran the full two-level
schedule both with the shipped kicks and with the kicks zeroed:

```
shipped (kicks 0.06 / 0.03):  100-level 0.891535,  final 0.894706
no kicks (0.0 / 0.0)       :  100-level 0.891887,  final 0.894363
```
