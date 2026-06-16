Uncorrected Adam answered the open question, and the answer is "only AdaGrad's flavour was poison." The
numbers: `n* = 50` on `d200_k5_s01` (tied with SGD at the grid floor), `56` on both `d500_k10` settings,
and `350` on `d10000_k50`. Scores `−5.64, −5.81, −5.81, −8.45`. Against SGD's `50 / 62 / 62 / 487` the
verdict is clear: Adam *matched* SGD where there was nothing left to win (the small problem, both at the
floor), edged it on the mid-size settings (`56` vs `62`), and — the decisive one — beat it cleanly at
high dimension, `350` vs `487` on `d10000_k50`. So forgetting plus momentum did not merely un-break
AdaGrad; the EMA preconditioner genuinely *helped* the geometry over bare GD, most where it mattered
most. That kills the "all adaptivity is poison" hypothesis: a *bounded*, forgetting, scale-normalized
denominator improves on SGD's bare step, because it accelerates the small support coordinates' escape
without the runaway damping AdaGrad suffered. And it tells me exactly where the remaining sample-complexity
is hiding. Adam still has one obviously sub-optimal knob for this problem — the second-moment memory.
With `β₂ = 0.999` the denominator `s_t` averages squared gradients over an effective window of roughly
`1/(1−β₂) = 1000` steps. On the diagonal-net the gradient scale on a support coordinate is *not*
stationary: it is near zero while the coordinate sits in the saddle, then surges as the coordinate
escapes, then settles at the interpolating value. A 1000-step memory *lags* that surge — when a support
coordinate finally takes off, its `s_t` is still dominated by the long stretch of near-zero gradients
before the escape, so the denominator is too small and the step on that coordinate is, briefly, too
large; conversely once it settles, `s_t` is still inflated by the surge and the step is too small. The
denominator is always tracking the past, never the present. So the natural next move is to *shorten the
second-moment memory*: drop `β₂` so the denominator responds to each coordinate's escape as it happens.

Let me reason about exactly how short, because this is the whole content of the rung — everything else
is held from the Adam I just measured. The effective averaging window is `1/(1−β₂)`. At `β₂ = 0.999`
that is 1000 steps, far longer than the duration of a single coordinate's saddle escape. Shorten it to
`β₂ = 0.95` and the window is `1/(1−0.95) = 20` steps — short enough that when a support coordinate's
gradient surges, `s_t` catches up within a couple of dozen iterations and the per-coordinate rate
re-normalizes to the *current* gradient scale, not the stale one. That is the responsive denominator
the non-stationary escape wants. The risk of going too short is the usual one: with a 20-step window the
squared-gradient estimate is noisier, so the denominator itself jitters, and a jittery denominator
injects extra noise into the step. But on this problem extra step-noise is not purely a cost — the whole
sparse bias is *driven* by noise, and the harness's Rademacher label noise is already the engine. A
slightly noisier, more responsive denominator keeps the noise temperature up while tracking the escape,
which is plausibly *better* for the sparse bias than the cold, smooth, lagging denominator at
`β₂ = 0.999`. So `β₂ = 0.95` is a bet that responsiveness-plus-temperature beats smoothness-plus-lag on
the diagonal-net — and it is testable against the Adam numbers directly.

Shortening `β₂` forces a paired change in `lr`, and I have to get the interaction right under the
*uncorrected* update, because that is what the harness runs. Recall the no-bias-correction subtlety from
the previous rung: at `t = 1` the step magnitude is `(1−β₁)/(1−β₂)^{1/2}` times the sign of the
gradient, because the moments are used raw, biased toward zero. At `β₂ = 0.999` that prefactor was
`0.1/0.0316 ≈ 3.16`; at `β₂ = 0.95` it is `0.1/sqrt(0.05) = 0.1/0.2236 ≈ 0.447`. So *shortening the
window shrinks the uncorrected early step by a factor of ~7* — the denominator `(1−β₂) g²` is much
larger at the start with a short window, so the raw ratio is much smaller. That is a real effect of
omitting the correction: with a long window the uncorrected start over-steps, with a short window it
under-steps, relative to the nominal `lr`. To compensate, the short-window configuration must *raise*
`lr` to keep the early escape fast — and so the baseline pairs `β₂ = 0.95` with `lr = 0.1` (double the
`lr = 0.05` used at `β₂ = 0.999`). This is not a free hyperparameter; it is the correction-omission
arithmetic demanding it. Doubling `lr` back to SGD's `0.1` while halving the window restores a healthy
early step size and lets the responsive denominator do its work for the rest of training. Everything
else is identical to the previous rung — `β₁ = 0.9`, `eps = 1e-6`, four zero-initialised EMA buffers,
the uncorrected `m_t / (sqrt(s_t)+eps)` update on both `u` and `v`. The harness still gives me only
`delta` and no noise-scale argument, so there is nothing else to adapt; the `eps` floor stays at `1e-6`,
below any healthy gradient RMS. The full scaffold module is in the answer.

There is a subtlety in *why* the gain should be largest on the `d500_k10` settings rather than spread
evenly, and it is worth spelling out because it is the precise prediction this rung makes. On the
`d200_k5` problem the sample requirement is already pinned at the grid floor of `50` — the search cannot
register any improvement below its smallest candidate, so a better optimizer simply ties there; there is
no headroom to win. On `d10000_k50` the binding constraint is different again: at extreme dimension and
sparsity the recovery threshold sits well inside the grid (`350`), and the dominant cost is getting the
fifty support coordinates to escape the saddle at all within the step budget — a *feasibility* pressure
that the early-step size (set by `lr`, which I am restoring to `0.1`) governs more than the
denominator's window length. It is only in the middle regime — `d500_k10`, where recovery happens at a
few dozen samples just above the floor — that the search has the resolution to resolve a handful of
samples *and* the escape is fast enough that the denominator's *responsiveness*, not its early
magnitude, is the marginal lever. That is exactly where shortening `β₂` should bite: the lag I am curing
is a fine-grained tracking error, and it shows up as a few-sample difference precisely where the problem
is neither floor-limited nor feasibility-limited. So a win that is concentrated on the two `d500_k10`
settings and absent at the extremes is not a weak result — it is the *signature* of the mechanism I am
claiming, and its absence (a flat tie everywhere, or a win at the extremes instead) would falsify it.

One more check on the geometry before I commit, because I want to be sure the short window is not
reintroducing the AdaGrad failure. AdaGrad's defect was *monotone accumulation* — the denominator only
grew, decaying the rate to zero. A short EMA window is the *opposite* failure mode in principle: it
forgets too fast and the denominator could drop, *raising* the rate uncontrollably. But on this problem
the denominator floor is the recent squared gradient, which on a converged support coordinate is the
interpolation residual scale (small but nonzero, kept alive by the label noise), and `eps` catches the
true zeros. So the short window does not blow up the rate; it makes it track. And critically, a short
window does *not* damp the support escape the way AdaGrad did, because it does not accumulate the
support coordinates' large gradients into an ever-growing penalty — it lets them surge, normalizes to
the surge, and releases. That is the responsive, non-damping adaptivity the diagonal-net rewards, and it
is the precise opposite of AdaGrad's compounding penalty.

The falsifiable expectation, against Adam's measured `50 / 62`-beating `56 / 56` and `350`. If the
responsiveness story holds, shortening `β₂` to `0.95` (with `lr` doubled to `0.1`) should recover at or
below Adam's numbers on every setting, with the gain concentrated where the non-stationary escape lag
hurt most. On the small `d200_k5` problem both are already at the grid floor of `50`, so I expect a tie
— nothing left to win there. On the mid-size `d500_k10` settings I expect `adam2` to *edge* Adam, from
`56` down toward the low fifties, because the responsive denominator tracks the support escape a little
tighter and the slightly hotter noise sharpens the sparse bias. The setting to watch is `d10000_k50`:
if the short window helps, `adam2` should hold or improve on Adam's `350`; if the short-window jitter
costs more than the responsiveness buys at extreme dimension, it could give a few samples back. My bet
is that `adam2` matches Adam on the small and large ends and wins on the mid settings — making it the
best rung overall by the thinnest of margins, a refinement of the Adam win rather than a new mechanism.
And that is the natural place to stop: the ladder has gone bare-GD-beats-bad-adaptivity, then
forgetting-adaptivity-beats-bare-GD, then responsive-forgetting-edges-it — the diagonal-net rewards a
bounded, responsive, scale-normalized step that rides the multiplicative escape and the label noise,
and `adam2` is the cleanest expression of that on the baseline grid. To beat it I would need a method
that *explicitly* shapes the sparse bias rather than tuning the preconditioner's window — a different
move entirely, and one the baseline family does not contain.
