The flat profile sat exactly at `2/3`, and the way it sat there told me what I have to overcome. Its
autoconvolution is the worst-case tent, realizing only `⟨t⟩/T = 1/3` in the one variable that governs the
score — `R = 2⟨t⟩/T` — the far corner from the box's `1/2`. The reward for moving is large and quantified:
a modestly flat-capped autoconvolution reaches into the high `0.8`s (a trapezoid with a cap as wide as its
ramp already scores `5/6`). But the reward is *not* aligned with any single height — growing a flat cap is
a coordinated reshaping of many heights at once, and the flat point is a plateau where no one-coordinate
nudge feels more than a sliver of the gain. So the only way forward is to introduce variation and let a
search find the coordinated non-flat profile. The two decisions are how to search and at what scale.

Scale first. The coordination requirement is the reason not to go straight to a long height vector: at `N`
in the hundreds a stochastic search hunting for a coherent spike-plus-shoulder profile wanders in a space
where the good region is a vanishingly thin sheet, the functional is highly non-convex, and translation
and reflection symmetry must be broken — all of which compounds at high dimension. Cost compounds too:
every score is an `O(N log N)` FFT, so an evaluation at `N = 500` costs roughly `65×` one at `20`, and
reshaping a long vector by one-coordinate moves needs orders of magnitude more iterations *and* has no good
shape to anchor on. The smart move is to find the *shape* at low resolution and lift it later. A hand-tuned
`20`-step profile already clears `0.88`, so I work at `N = 20`: `2N+1 = 41` nodes, coarse enough to canvas
in seconds, long enough for real internal structure — though I already suspect `41` nodes cannot render a
very wide plateau with a very sharp shoulder.

Now the search. The obvious first thing is greedy hill-climbing: perturb a height, keep improving moves,
revert otherwise. I run it to see *where* it crawls. From a flat start it stalls around `0.81`; from a
smooth single-bump start it stalls *lower*, around `0.75`. Both are trapped far short of the `0.88` band.
That the bump — chosen precisely because a concentrated profile *should* sit near the good shape — stalls
lower than the flat start is the decisive evidence: the good basin is not simply downhill from a
well-chosen initialization, so no amount of clever seeding plus greedy refinement gets there. The
coordinated support I am after (a spike, a shaped shoulder, genuine zeros between) is separated from both
simple starts by ridges that go down before they go up. Downhill tolerance, not a better starting point,
is the load-bearing ingredient — which is exactly simulated annealing: propose a one-height perturbation,
take it if it improves `R`, take it anyway with Metropolis probability `exp((R_new − R_cur)/T)` if it
worsens, under a temperature cooled geometrically. Hot early to shake loose from shallow basins, cold late
to settle into the best one found.

Three choices come from the geometry of this objective. The *form of the perturbation*: heights are
non-negative and, if the rung-1 reading is right, span a wide dynamic range — a tall spike plus a long
shoulder of much smaller values. A fixed additive Gaussian kick would swamp the small shoulder heights and
barely move the spike, so I make the kick multiplicative in scale: perturb `v_j` by `N(0, σ(0.1 + v_j))`,
reflecting negatives back, so spike and shoulder move on comparable *relative* terms — the right invariance
for a scale-free objective, with the `0.1` floor keeping a near-zero height revivable. *What to anneal on*:
because `R ∈ [0,1]` with small well-scaled changes, I anneal directly on `R` with `T` of order `10^{-3}`
down to a tiny floor — at `T = 10^{-3}` a downhill move of size `0.001` is accepted with probability
`e^{-1} ≈ 0.37`, frequent enough to cross ridges; near `10^{-6}` essentially never, so the late run is
nearly greedy. And *restarts*, since the landscape has many basins of differing quality: many independent
runs from a Gaussian single-bump seed and from pure random heights, keeping the best profile any reaches.

Alongside the annealing I want a gradient method carrying the bulk of the descent, because a gradient step
is a *coordinated* move over all twenty heights at once — exactly the joint reshaping the flat point could
not feel, and what single-coordinate annealing assembles only slowly and by luck. The hard `max` in the
denominator blocks a naive gradient, so I smooth it: `‖f*f‖_∞ = max_j L_j` becomes the softmax
`B(β) = m + β^{-1} log Σ_j e^{β(L_j − m)}` (with `m = max_j L_j` subtracted for stability), differentiable
and amenable to L-BFGS. The approximation size dictates the schedule: since the argmax contributes `e^0 = 1`
among `41` terms, the softmax overestimates the peak by at most `log(41)/β ≈ 3.71/β`. At `β = 5` that
overshoot can be `0.74` — a broad, forgiving surrogate that lets early L-BFGS steps move over a smooth
landscape; at `β = 6000` it is `≈ 0.0006` — a faithful stand-in whose optimum is the real ratio's. So I run
a ladder `5 → 15 → 40 → 120 → 400 → 1500 → 6000`, each solve warm-started from the last. One subtlety for
later: a flat-capped autoconvolution clusters many nodes near the peak, so the overshoot is closer to
`log(#tied nodes)/β` — as the cap flattens, `β` must sharpen to stay faithful. At `41` nodes this is mild.

So the pipeline has a deliberate division of labor. The workhorse is the L-BFGS ladder run as a multistart
— from roughly a hundred seeds, climb the whole `β` ladder, keep the best — each step a quasi-Newton move
reshaping all twenty heights together against the smooth surrogate. Then, on the best candidate, a long
simulated-annealing polish directly on the *exact* `R`: it crosses the residual non-smooth ridges the
softmax rounded over, and settles the sparse support by nudging near-zero shoulder heights down to exact
zero. Neither half alone suffices: pure L-BFGS sticks at the surrogate's optimum, and pure single-coordinate
annealing stalls in the low `0.8`s, as measured. Whether the optimizer drives heights to *exactly zero* is
a real signal I watch for — a spike plus a shaped shoulder with genuine gaps would confirm the
concentrated-yet-shaped input the smoothing argument predicted.

Annealing clears the tent floor easily and the search lands at `R = 0.884823`, `+0.218` off the floor,
with exactly the predicted structure: `v_0, v_1` and `v_4…v_7` driven to zero, a tall spike `v_2 = 1.0`,
and a shaped shoulder `v_8…v_19` tapering from `~0.5` toward zero. Reading the autoconvolution confirms the
mechanism: on the order of fourteen of the forty-one nodes sit within one percent of the peak — a genuine
plateau, not the single apex of the tent — with steep sides and a short tail. In the layer-cake variable
this pushes `⟨t⟩/T` from `1/3` to about `0.442`, most of the way toward the box's `1/2`. The whole `+0.218`
is bought by turning one apex node into a fourteen-node plateau: a spike to build the convolution's height
fast at the plateau's edges, a tapered shoulder to hold the cap level, the useless near-zero pieces cleared
away. One feature is a free symmetry, not a defect — the profile is lopsided (spike near the front, shoulder
trailing), but `R` is invariant under reversing and translating the support, so every optimum is an orbit
of reflected and shifted copies scoring identically and the search simply lands on an arbitrary member.

The value lands `0.0044` below the published twenty-step `0.88922` — the same band, honestly short of it,
and the layer-cake says why. My plateau is about a third of the support wide; to push `⟨t⟩/T` closer to
`1/2` I need `μ(t)` to hold near its base value further up toward `T` and then fall off *sharply*, but a
sharp fall-off is a steep shoulder, and at twenty pieces the shoulder can only descend in a few coarse steps
before it hits zero. The grid cannot render both a wide plateau and a sharp shoulder at once — both my
compromise and the published one are pressed against the same forty-one-node ceiling. If resolution is the
wall and the search idea is sound, then lifting this same shape onto a finer grid — room for both a wider
plateau and a sharper shoulder — should keep raising `R` past `0.885` toward the high `0.89`s with no new
shape idea; if a finer grid did *not* help, resolution is not the wall and I go back to the shape. That is
the opening for the next step: refine this coarse profile onto a much finer grid and let a gradient method
carve structure twenty heights cannot represent.
