The hierarchical lifts kept paying — `0.884823` at twenty pieces, `0.894706` at five hundred — and when I
stopped the gradient was still moving, the per-lift gains diminishing but strictly positive. The
falsifiable prediction was that if I am refining a fixed spike-and-shoulder shape whose ceiling is the
smooth-basin frontier near `0.90`, then lifting once more to a few thousand pieces and grinding much longer
should push into the low `0.90`s and then *flatten*. I test that here. The published step-function frontier
is concrete — the `575`-piece `0.901564` and the `539`-piece `~0.9016` — and both live in the range a few
thousand well-optimized pieces can render, so the target is to push up to and, if I can, a hair through that
band, knowing the record near `0.96` sits far above and was bought by a different kind of search.

The lift is the same free operation: upscale the five-hundred-piece profile `×4` to `N = 2000` (same
function dilated, `R` unchanged to ten digits), kick the degenerate flat-block plateau, and refine with
`β`-annealed Adam on the analytic FFT gradient. `×4` is a deliberate balance: two thousand renders the
`539`/`575`-piece frontier shapes with cells to spare — past the point where the grid binds, which the
five-hundred diagnosis said was already the case — while ten thousand would cost `~2.5×` more per step and
land in the same place, since my whole reading is that the `0.90` ceiling is a property of the *shape
family*, not the grid. Two thousand is the smallest resolution that comfortably clears the frontier's
rendering needs while keeping a forty-thousand-step grind inside a two-minute budget.

Three things are forced by the finer problem. First, the run has to be long: at two thousand the optimum is
a finer, more irregular profile with four times as many coordinates to bring into a coherent arrangement,
and the gradient keeps finding small improvements for tens of thousands of steps. So I budget a
`~40000`-iteration final pass — affordable only because one step at two thousand costs about `4.9×` a step
at five hundred (the `N log N` ratio), putting the whole ladder at about `130` seconds.

Second, the `β` schedule has to be pushed *much* sharper at the end, and the reason is in the gradient, not
the level. In absolute terms the softmax is faithful even at moderate `β` — with `4001` nodes the overshoot
is at most `log(4001)/β ≈ 8.3/β`, under `10^{-4}` by `β ≈ 80N`. But its job in the gradient is to control
the true peak through the softmax weight `w_j/Z`, which should concentrate on the genuine argmax. A flat
wide cap is precisely a cluster of many *nearly tied* top nodes, and the weight smears across all of them
unless `β` times their tiny spacing is of order one. As the optimizer flattens the cap the spacing shrinks,
so a `β` adequate at five hundred pieces goes too soft — the peak-control gradient blurs, the cap can tilt
or a spike creep up while the surrogate barely notices, and true `R` lags. The fix is to anneal `β` up to
several hundred times `N` — about `400N` in the grind, sharper in the polish — so the softmax resolves the
crowded near-peak nodes; the flatter the cap, the sharper `β`, which is why the `~123N` ceiling at five
hundred becomes `~400N`–`800N` here. This is also why I sharpen a softmax rather than optimizing the hard
`max`: the subgradient of `max_j L_j` is one-hot on whichever node is currently tallest, so on a nearly-tied
plateau the tallest node hops between steps and the optimizer chatters, never settling the plateau as a
whole. The softmax's smeared weight controls the entire cap coherently at soft `β` and hands control to the
true peak as it sharpens.

Third, periodic kicks *during* the long run, not just at the upscale. A single kick unsticks the initial
plateau, but over tens of thousands of steps the optimizer can settle into a *shallow* basin and stop well
before carving all the fine structure the resolution allows. A small multiplicative kick every few thousand
steps acts as a mild restart, its spacing sized so each lands after the current basin is exhausted but
before many steps are wasted in it. I shrink `κ` as the run sharpens (`~0.008` in the grind toward nothing
in the polish) so the late phase is pure refinement, and because a kick can temporarily lower `R` the
best-true-`R` bookkeeping earns its keep — the best vector across the whole schedule is kept, so a kick can
never cost me the good profile. This periodic-kick mechanism is the one genuinely new ingredient, there
specifically because the run is now long enough to stall mid-way, which it was not at five hundred.

So the endpoint is a short ladder of passes at two thousand: a moderate-`β` kicked *reorganize* pass to let
the lifted shape find its new plateau width; the long high-`β` kicked *grind* where the irregular fine
structure is carved and the number is made; and a low-lr sharpest-`β` *polish*. The knobs move in lockstep
— `β` ceiling sharpens (`~80N → ~200N → ~400N`, sharper in polish), learning rate decays (`~0.006 → 0.003
→ 0.002 → 0.0008`), kick shrinks to zero. Splitting reorganize from grind is deliberate: right after the
`×4` lift the shape is far from settled, and sharp-`β` steps are stiff and expensive, so a cheaper
moderate-`β` pass does the bulk reshaping first and the sharp grind is reserved for when there is genuine
fine structure to carve — the same coarse-then-fine logic as annealing `β` within a pass, lifted to the
sequence of passes.

The prediction holds closely. The reorganize pass clears the five-hundred-piece value comfortably, around
`0.899`, and the long sharp-`β` grind produces `R = 0.901804`, `+0.0071` over the five-hundred rung — *at
and slightly above* the published frontier, matching the `539`-step `~0.9016` and exceeding the `575`-step
`0.901564`, with two thousand pieces in about `130` seconds. The layer-cake variable `⟨t⟩/T = R/2` reaches
about `0.451` — right at the `~0.45` projected as the shape's ceiling, and *approaching* it rather than
blowing past. The returned profile has the structure the frontier is known for: genuinely irregular and
sparse, roughly a third of its heights effectively zero, a spike on the order of thirty times the shoulder,
a jagged spike-and-shoulder with a wide faintly-rippled flat cap. That is a consistent continuation of
everything below — the coarse twenty-piece profile was already about thirty percent zeros with a spike some
seventeen times its smallest shoulder, and across the lifts the sparsity fraction *held* (zeros lift to
zeros) while the spike *sharpened* seventeen-fold to thirty-fold as the cap flattened. The sharpening spike
is what the layer-cake asks for: a taller, narrower spike builds the autoconvolution's height faster at the
edges, steepening the sides so `μ(t)` holds near its base value further up toward `T` — exactly the
intermediate object the first rung predicted from the smoothing map, now resolved a thousandfold higher.

The roughly one-third zeros are a genuine optimum, not a clipping artifact. Clipping to non-negative could
in principle pin heights at zero and freeze them, but the gradient at a zeroed height is `2` times the
correlation of the node-gradient with the rest of the profile, generally non-zero, so Adam can push a
zeroed height back positive on any later step if that helps. The heights that stay at zero stay because the
optimizer keeps choosing zero for them across kicks and sharpening `β` — a real feature, the same signal
the coarse anneal gave, now confirmed at high resolution by a different optimizer.

And here the ladder stops, on a test rather than a mood. If I were still grid-limited, lifting further would
keep raising `R` and `⟨t⟩/T` toward `0.5`; if I have hit the shape's floor, the value pins below `0.5` and
the marginal return collapses. Both signatures point the same way: `⟨t⟩/T` landed at `0.451`, essentially
*at* the projected `~0.45` and not creeping past, and the per-lift gains have decayed to a thin slice
(`0.884 → 0.895 → 0.902`, each step smaller). So `0.9018` is a *shape* limit, not a resolution limit — the
smooth spike-and-shoulder basin every careful local optimizer in this problem converges to, floored near
`0.90`. The record near `0.96` was reached by a large-scale evolutionary / test-time search over
*deliberately irregular* many-plateau functions with tens of thousands of pieces, whose `μ(t)` stays
near-box much higher up. Traveling from my single smooth plateau to that many-plateau top requires first
*breaking* the plateau — momentarily lowering `⟨t⟩/T` and `R` — and rebuilding it, an up-front downhill cost
no gradient method and no mild-restart annealer will pay: my kicks are small jostles of a spike-plus-plateau
that stay a spike-plus-plateau, exploring *within* the basin. So the honest number is `0.901804`, and the
gap from there to `0.96102` is the measure of how much of the second autocorrelation inequality is still
open. There is no finale in which this constructor grinds its way to `0.96`; that requires a fundamentally
different search.
