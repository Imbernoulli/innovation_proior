The coarse annealing gave a twenty-piece profile at `0.884823`, just under the published `0.88922`, with a
specific diagnosis: the shape is right — a tall spike, a shaped shoulder, several heights at zero, an
autoconvolution with a flat cap about fourteen of its forty-one nodes wide — but twenty pieces cannot render
both a wide plateau and a sharp shoulder, so the grid caps how high `⟨t⟩/T` can climb. The prediction was
that lifting this same shape onto a finer grid, with no new shape idea, should let the plateau widen and the
shoulder steepen and push `R` into the high `0.89`s. I test that by *keeping* the coarse shape and handing
it more resolution, rather than searching a long vector from scratch — which the coarse run showed is
hopeless.

The lift is free: replacing each of the twenty heights by `k` copies produces a `20k`-piece function that
is literally the same function dilated, and `R` is dilation-invariant, so `R(v') = R(v)` exactly — lifting
by `k = 2, 5, 10` returns `0.8848227841` to ten digits. But the lifted point itself is a *degenerate
plateau*: every block of `k` identical copies is flat, and the within-block gradient vanishes by symmetry.
The lifted profile is invariant under reflecting each block, `R` is reflection-invariant, and "make the
left copy larger, the right smaller" is *odd* under that reflection, so its directional derivative equals
its own negative and is zero. The new degrees of freedom the lift created are switched off at the lifted
point; something has to switch them on.

Before committing I rule out the obvious alternative — skip the annealing, run gradient ascent on five
hundred heights from a smooth bump seed. The coarse run already answers it: a gradient method is a
hill-climber, and hill-climbing from a bump seed stalled in the mid-`0.7`s because the good support is
separated from any smooth start by ridges that descend before they climb. Adam from a bump seed at five
hundred pieces finds that same poor basin, faster. What makes the lift work is not the gradient but the
object being lifted — the downhill-tolerantly searched support already on the far side of those ridges. The
hierarchy is a necessity with a clean division of labor: the anneal supplies a support gradient cannot
reach, the lift carries it to a resolution where gradient can refine it. Skipping either half fails.

So I need to break the within-block symmetry and move all the new coordinates together — the second is the
binding constraint, since single-coordinate annealing is hopeless at `N` in the hundreds (each evaluation
dozens of times costlier, a coherent shoulder reshape now spanning a hundred coordinates a one-at-a-time
move can only align by luck). I compute the gradient of `R` and smooth the `max` as before. With
`A = ‖f*f‖_2^2`, `C = ‖f*f‖_1`, `B(β)` the softmax peak, surrogate `Q = A/(BC)`, the quotient rule gives
`∂Q/∂L_j` from three local pieces: `∂A/∂L_j = ⅓(4L_j + L_{j+1} + L_{j−1})`, `∂C/∂L_j = 1` interior, and
`∂B/∂L_j = e^{β(L_j−m)}/Z` — the softmax weight, peaked at the argmax and hardening to an indicator as
`β → ∞`. Read together the three terms *are* the flatten-the-cap instruction: the energy term pushes all
high nodes up, the peak term is a penalty concentrated on the current tallest node pushing it down, the
mass term subtracts uniformly — raise the sub-peak nodes toward the peak while penalizing the lone highest,
widen the plateau without letting a single spike run away. The last link chains through the self-convolution:
`c_k = Σ_n v_n v_{k−n}` gives `∂c_k/∂v_p = 2 v_{k−p}`, so `∂Q/∂v_p = 2 Σ_k (∂Q/∂c_k) v_{k−p}` — one more
convolution, the whole gradient a handful of FFTs at `O(N log N)`, as cheap as one evaluation and agreeing
with finite differences to `~10^{-10}`. One Adam step then advances all five hundred coordinates coherently,
where the annealer would need millions of evaluations for the same progress; at this resolution the gradient
is the only affordable way to move coordinately at all.

The optimizer riding it is Adam, because of the dynamic range already visible in the coarse profile —
heights spanning more than an order of magnitude, spike at `1.0` down to shoulder around `0.05`, and the
lift sharpens the spread further. A fixed-step gradient ascent is wrong: a step large enough to move the
spike blows the thin shoulder negative (then clipped to zero, destroyed), a step safe for the shoulder
crawls on the spike. Adam's per-coordinate scaling (`b1 = 0.9`, `b2 = 0.999`) advances spike and shoulder
on comparable relative terms — the same scale-free invariance the coarse multiplicative perturbation gave.
After each step I clip to non-negative and track the best *true* `R`, not the surrogate: since `B ≥` the
true peak, `Q ≤ R` and the surrogate's maximizer sits slightly off the real one, so I evaluate the exact
ratio each step and return the best-`R` vector.

The `β` ceiling has to grow with resolution: the overshoot is `log(2N+1)/β`, so `β ≈ 83N` keeps it under
`6·10^{-4}` at `N = 100` and `β ≈ 123N` under `10^{-4}` at `N = 500`. And the flat-cap subtlety now bites
— the flatter the cap, the more nodes cluster near the peak, so the overshoot approaches `log(#tied
nodes)/β` and `β` must be pushed sharper to keep `B` faithful, otherwise the optimizer chases a blurred peak
and true `R` lags. I anneal `β` geometrically soft-to-sharp within each pass and run two passes per level.
The ingredient that switches the degenerate plateau on is a small multiplicative kick `v ↦ |v(1 + ε ξ)|`
right after each upscale (`ε ≈ 0.06` at the first lift, `~0.03` at the finer one — smaller because the
carved structure is more delicate). The two passes have distinct jobs: the first carries the kick and a
moderate `β`, letting the freshly lifted shape reorganize over a still-smooth surrogate; the second, no kick
and sharper `β`, polishes against a nearly-faithful `max`, tightening the plateau and steepening the
shoulder.

The `×5`-then-`×5` factoring balances two extremes. A single `×25` jump would carve all the within-block
structure from a crude blocky init — nearly optimizing five hundred heights from scratch, discarding the
graded refinement the hierarchy exists for. Many tiny `×2` lifts pay the fixed cost of a kick and two Adam
passes while barely adding degrees of freedom. `×5` gives four new within-block degrees of freedom per
piece — enough for a kick to find real asymmetric structure and the gradient to exploit it — and two lifts
span twenty-to-five-hundred in the graded way intended.

The prediction comes true. The lift to `100` clears the twenty-piece value comfortably, around `0.89`; the
lift to `500` settles at `R = 0.894706`, `+0.0099` over the coarse rung and just under the fifty-step
`0.89628`. In the layer-cake variable `⟨t⟩/T` moves from `0.442` to `0.4474` — a small real step further
toward `1/2`, the "hold higher, fall faster" the finer grid was supposed to buy. This does *not* reach the
`~0.9016` of the best published `~575`-step constructions, and the reason is no longer resolution: at five
hundred pieces the autoconvolution has a thousand nodes, plenty of grid for both plateau and shoulder. The
residual gap is that pushing `⟨t⟩/T` the last sliver requires the shoulder to take a specific irregular,
non-monotone profile, and a few thousand smooth Adam steps carve most of it but not the last — the far
larger searches spent vastly more compute on exactly that fine irregular structure. The support survived
every upscale (a lifted zero is `k` zeros, so the coarse gaps are preserved and the refined profile is a
finer rendering of the same spike-plus-shoulder-plus-zeros support), and the per-lift movement is monotone
and decelerating — the signature of polishing a shape already essentially right.

The gradient is still moving at `500`, which is the opening for the endpoint, and I can make it falsifiable.
If I am refining a fixed shape whose ceiling is the smooth spike-and-shoulder family sitting around `0.90`
at the published `~575`-step frontier, then lifting to a few thousand pieces and grinding much longer should
push into the low `0.90`s and then *flatten* — approach roughly `0.90`, match the best published
step-function results, and stop climbing no matter how much more resolution I add, with `⟨t⟩/T` creeping
toward but not past `~0.45`. The honest expectation is the saturating one: a frontier this ladder reaches
and cannot exceed without a different kind of search.
