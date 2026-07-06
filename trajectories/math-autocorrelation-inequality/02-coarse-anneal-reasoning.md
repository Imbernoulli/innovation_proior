The flat profile sat exactly at `2/3`, and the way it sat there told me precisely what I have to overcome.
Its autoconvolution is the worst-case tent, and in the one variable that governs the score — the μ-weighted
mean level `⟨t⟩/T`, where `R = 2⟨t⟩/T` — the flat profile realizes only `1/3`, the far corner from the box's
`1/2`. The reward for moving is large and I quantified it: a modestly flat-capped autoconvolution reaches
into the high `0.8`s (a trapezoid with a cap as wide as its ramp already scores `5/6`). But I also learned
that the reward is *not* aligned with any single height — growing a flat cap on `f*f` is a coordinated
reshaping of many heights at once, and the flat point is a near-flat plateau where no one-coordinate nudge
feels more than a sliver of the gain. So the only way forward is to introduce *variation* among the heights
and let a search find the coordinated, non-flat profile that bends the autoconvolution away from the tent.
The two decisions are how to search and at what scale.

Scale decides everything else, so I fix it first, and I deliberately resist the temptation to go straight to
a long height vector. The coordination requirement from the previous rung is the reason: the useful move is a
joint reshaping across many coordinates, and the number of distinct coordinated reshapes grows explosively
with the vector length. At `N` in the hundreds, a stochastic search that perturbs heights and hopes to
stumble onto a coherent spike-plus-shoulder profile is hunting in a space where the good region is a
vanishingly thin sheet; it wanders for an enormous time before finding anything. The functional is also
highly non-convex, with many local optima and a great deal of translation and reflection symmetry to break,
which only compounds the problem at high dimension. The smart move is to find the *shape* at low resolution,
where the vector is short enough to canvas thoroughly, and only later lift that shape onto a finer grid. The
published landmark points exactly here — a hand-tuned profile of only twenty steps already clears `0.88` —
so I work at `N = 20`: coarse enough that a stochastic search can explore the shape space exhaustively, long
enough that the autoconvolution has real internal structure to exploit. In layer-cake terms, `N = 20` gives
me `2N + 1 = 41` nodes on the autoconvolution, so a flat cap has to be built out of a handful of consecutive
top nodes tied near the peak — buildable at this resolution, though I already suspect only coarsely, since
`41` nodes cannot render a very wide plateau with a very sharp shoulder.

I want to be concrete about why *not* to just start large, because "small first" is a real bet and I should
be able to defend the arithmetic. Suppose I went straight to `N = 500`. Every evaluation is an FFT
self-convolution, `O(N log N)`, so a single score at `500` costs roughly `500·log 500 / (20·log 20) ≈ 65`
times a score at `20` — the per-iteration price alone is nearly two orders of magnitude. Worse, the search
is not a hill; it is a landscape whose number of distinct local optima grows with dimension, and a
single-coordinate stochastic move assembles a coherent spike-plus-shoulder only by chance. To reshape a
`500`-vector into a coordinated profile by one-coordinate-at-a-time moves, the search has to get hundreds of
heights into the right relative arrangement before cooling freezes it, which needs orders of magnitude more
iterations than at `20` — and it would be doing this *from scratch*, with no good shape to anchor on. The
budget simply does not close: expensive iterations, exponentially more of them needed, and no warm start.
Whereas at `N = 20` the whole space is short enough that a hundred-odd restarts plus a long polish run in
seconds to a couple of minutes and genuinely canvas the shape. So I buy the shape cheaply here and defer
resolution to a later lift, rather than pay the high-dimensional search cost with nothing to guide it.

Now the search itself, and here I want to be empirical rather than assume. The obvious first thing to try is
a greedy hill-climb: perturb a height, keep the change if `R` improves, otherwise revert. From the previous
rung I know the flat point has a whisper of gradient, so greedy will not sit at `2/3` — it will crawl. The
question is *where it crawls to*, so I run it. From a flat start it climbs into the low `0.8`s and stalls
there, around `0.81`; from a smooth single-bump start it stalls even lower, around `0.75`. Both are trapped
far short of the `0.88` band where the known twenty-step constructions live. That is the concrete obstruction:
the landscape is riddled with poor local optima, and a climber that only ever accepts improving moves parks
itself at the first ridge it reaches and cannot get across the valley that separates its poor basin from the
good one. To cross those valleys I have to be willing to accept moves that make `R` temporarily *worse* —
which is exactly what simulated annealing does. I propose a perturbation to one height; if it improves `R` I
take it, and if it worsens `R` I take it anyway with Metropolis probability `exp((R_new − R_cur)/T)` under a
temperature `T` that I cool geometrically over the run. Hot early, so the search wanders freely and shakes
loose from whichever shallow basin it is in; cold late, so it settles into the best basin it has found. The
whole bet is that with enough downhill-tolerant wandering it discovers the coordinated spike-plus-shoulder
profile whose autoconvolution has a flat cap and steep sides — the shape my one-variable analysis says lives
well above `2/3`.

The two greedy stall numbers are worth reading against each other, because together they rule out the
cheaper fix I would otherwise reach for. The bump seed, which I chose precisely because a concentrated
profile *should* be near the good shape, greedily stalls *lower* (`~0.75`) than the flat start (`~0.81`).
If the good basin were simply downhill from a well-chosen initialization, the bump would have climbed higher,
not lower — so the target profile is not reachable by hill-climbing from either simple start, and no amount
of clever initialization plus greedy refinement gets me there. The coordinated support I am after — a spike,
a shaped shoulder, and genuine zeros between them — is separated from both the flat and the bump by ridges
that go *down before they go up*. That is the decisive evidence that downhill tolerance, not a better
starting point, is the load-bearing ingredient, and it also tells me the multistart has to be paired with a
mechanism that accepts worsening moves rather than being just many independent greedy climbs.

Three design choices come straight from the geometry of this objective, and each one is a real decision, not
a default. The first is the *form of the perturbation*. Heights are non-negative and, if my rung-1 reading is
right, span a wide dynamic range — the good profile is a tall spike or two plus a long shoulder of much
smaller values, not a uniform spread. A single additive Gaussian kick of fixed size would be wrong on both
ends: far too coarse for the small shoulder heights (it would swamp them) and far too timid for the tall
spike (it would barely move it). So I make the kick *multiplicative in scale*: I perturb a chosen height
`v_j` by `N(0, σ(0.1 + v_j))` and reflect any negative result back to non-negativity, so a tall spike and a
thin shoulder value are each adjusted on comparable *relative* terms. The `0.1` floor keeps a height that has
been driven near zero from freezing permanently — it can still be revived — while the proportional part gives
the spike room to move. This is the right invariance for a scale-free objective. The second choice is *what
to anneal on*. Because `R` is already bounded in `[0, 1]` and its changes under a single height perturbation
are small and well-scaled (fractions of a percent), I do not need to take a log or rescale the objective — I
can anneal directly on `R` with a temperature of order `10^{-3}` down to a tiny floor, and the Metropolis
acceptance behaves sanely: at `T = 10^{-3}` a downhill move of size `0.001` is accepted with probability
`e^{-1} ≈ 0.37`, frequent enough early to cross ridges, and as `T → 10^{-6}` such a move is essentially never
accepted, so the late run is nearly greedy. I shrink the perturbation scale alongside the temperature so the
late run makes fine adjustments to a settled shape rather than large jumps. The third choice is *restarts*.
A single annealing run from one seed can still get trapped, because the landscape has many basins of
differing quality, so I run many independent restarts and keep the best profile any of them ever reaches.
Some restarts start from a smooth single-bump (Gaussian) seed — the educated guess, since a unimodal
autoconvolution with a flat cap is most plausibly produced by a *concentrated* height profile, so starting
concentrated should land me in the good basin faster — and some from pure random heights, to avoid baking in
that guess and to sample basins the bump seed would never reach.

Alongside the annealing I want a gradient method carrying the bulk of the descent, because a gradient step
is a *coordinated* move over all twenty heights at once — exactly the joint reshaping the previous rung
identified as the only thing that raises `R`, and exactly what single-coordinate annealing can only assemble
slowly and by luck. But the hard `max` in the denominator blocks a naive gradient, so I smooth it. `‖f*f‖_∞ = max_j L_j` is not differentiable at the peak, so I
replace it by the softmax `B(β) = m + β^{-1} log Σ_j e^{β(L_j − m)}` with `m = max_j L_j` subtracted for
numerical stability, which makes the whole objective differentiable and amenable to L-BFGS. The size of the
approximation is worth pinning down, because it dictates the schedule: since one term of the sum (the
argmax) is `e^0 = 1` and there are `2N + 1 = 41` terms, `m ≤ B ≤ m + log(41)/β`, i.e. the softmax
*overestimates* the true peak by at most `log(41)/β ≈ 3.71/β`. At a soft `β = 5` that overshoot can be as
large as `0.74` — the surrogate is a broad, forgiving approximation that lets early L-BFGS steps move over a
smooth landscape without getting caught on the exact peak. At a sharp `β = 6000` the overshoot is at most
`≈ 0.0006`, so the surrogate is a faithful stand-in for the true `max` and its optimum is the optimum of the
real ratio. That is why I run a *ladder* of increasing sharpness — `5 → 15 → 40 → 120 → 400 → 1500 → 6000` —
starting broad and ending faithful, each L-BFGS solve warm-started from the last: the early rungs find the
right neighborhood, the late rungs polish against a nearly-exact objective. There is a subtlety I keep in
mind for later — a *flat-capped* autoconvolution has many nodes clustered near the peak, so the softmax sum
is dominated by more than one term and the overshoot is closer to `log(#tied nodes)/β` than `log(41)/β`; as
the search succeeds and the cap flattens, the surrogate needs a sharper `β` to stay faithful. At `N = 20`
with only `41` nodes this is mild, but I note it.

This gives the pipeline its actual two-stage shape, and the division of labor is deliberate. The workhorse
is the L-BFGS ladder run as a *multistart*: from each of roughly a hundred seeds — alternating the bump
guess and pure random heights — I climb the whole `β` ladder to a candidate profile, and keep the best score
any seed ever reaches. This is what finds the good basin, because each L-BFGS step is a quasi-Newton move
that reshapes all twenty heights together against the smooth surrogate — cheap, coherent, and repeated from
many starting points so no single bad basin dominates. Then, on top of the best candidate the multistart
found, I run a long simulated-annealing polish directly on the *exact* `R`, not the surrogate. The polish
does the two things gradient descent on a smoothed objective cannot: it crosses the residual non-smooth
ridges that the softmax rounded over, and it settles the sparse support — nudging the near-zero shoulder
heights down until they sit at exact zero — by tolerating the small downhill steps such consolidation
requires. So L-BFGS supplies the coordinated descent to the basin, and annealing supplies the downhill-
tolerant ridge-crossing and support-cleanup that a smooth gradient method structurally cannot. Neither alone
would do it: pure L-BFGS gets stuck at the surrogate's optimum and cannot cross the exact objective's
ridges, and pure single-coordinate annealing, as I measured, stalls in the low `0.8`s.

One thing I want to watch throughout is whether the optimizer drives some heights to *exactly zero*, because
that would be a real signal and not noise. If the best twenty-piece profile turns out to use only a subset
of its pieces — a spike plus a shaped shoulder with genuine gaps of zero between them — it tells me the gains
over `2/3` come from a specific *sparse, asymmetric* support, not from spreading mass evenly, which is
exactly the "concentrated-yet-shaped input" my rung-1 smoothing argument predicted. If instead the profile
came out dense and smooth, that prediction would be wrong and I would have to rethink the whole spike-plus-
shoulder picture.

What I expect, then, is that the annealing clears the flat-triangle floor easily — that is the entire reason
to accept downhill moves — and climbs into the high `0.88`s, the band where the known twenty-step
constructions live, and that the converged profile is genuinely sparse. Both hold. The search lands at
`R = 0.884823`, a jump of `+0.218` off the `0.6667` floor, and the profile it converges to is exactly the
predicted structure: several heights (`v_0, v_1` and `v_4…v_7`) driven to zero, a tall spike at `v_2 = 1.0`,
and a shaped shoulder from `v_8` to `v_19` tapering from about `0.5` down toward zero — a sparse, asymmetric
support, not all twenty pieces uniformly. Reading the autoconvolution confirms the mechanism directly: its
node values show a broad flat cap — on the order of fourteen of the forty-one nodes sit within one percent of
the peak, a genuine plateau rather than the single-point apex of the tent — with steep sides and a short
tail. In the one variable that matters, this profile has pushed the normalized mean level `⟨t⟩/T` from the
tent's `1/3` to about `0.442` (since `R = 2⟨t⟩/T`), most of the way toward the box's `1/2`. That is the
coordinated flattening the flat profile could not reach, achieved exactly by tolerating downhill moves to
cross the valleys a greedy climber stalled in at `0.81`.

I can read that score two independent ways as a check that I understand *why* it is what it is, rather than
having stumbled onto a number. Directly, the norms give `L_2^2 / (L_∞ · L_1) = 0.884823`. Through the
layer-cake, the same value should be `2⟨t⟩/T` with `⟨t⟩/T ≈ 0.442`, and `2 · 0.442 = 0.884` matches — so the
gain really is the μ-weighted mean level climbing from the tent's `1/3` toward the box's `1/2`, exactly the
mechanism my one-variable model predicted, and not some accidental coincidence of the three norms. Reading
the autoconvolution's nodes makes the mechanism physical: the tent had a single apex node at the top, and
this profile has widened that top into a plateau of roughly fourteen tied nodes, with the shoulder heights
`v_8…v_19` doing the work of holding the cap level while the spike `v_2` builds the convolution's height
fast at the plateau's edges. That is the concentrated-yet-shaped input the previous rung argued for, made
concrete: a spike to raise `f*f` quickly and a tapered shoulder to keep the cap from sagging, with the
useless near-zero pieces cleared out of the way. The whole `+0.218` over the floor is bought by turning one
apex node into a fourteen-node plateau.

One feature of the converged profile initially looks like a bug and is actually a symmetry, and noticing it
reassures me the search is behaving. The autoconvolution's plateau is not centered — its peak sits well off
the middle of the support — and the height profile itself is lopsided, spike near the front, shoulder trailing
behind. That is expected: `R` is invariant under reversing the height vector (`v_n → v_{N-1-n}` gives the
mirror-image function with the same autoconvolution up to reflection) and under translating the support, so
every optimum is really a whole orbit of reflected and shifted copies, all scoring identically. The search
has no reason to prefer the symmetric representative and simply lands on one arbitrary member of the orbit —
here the front-loaded one. This is the "symmetry to break" I worried about when choosing the scale: it does
not cost me anything at the objective level (all representatives score the same), but it does mean the raw
height vector is not directly comparable across restarts, and it is one more reason the landscape has many
equivalent basins rather than one clean optimum. Confirming that the asymmetry is a free symmetry and not a
defect lets me trust the `0.884823` as a genuine near-best-at-this-resolution value.

The value lands `0.0044` below the published twenty-step `0.88922` — the same band, and honestly short of it,
because a short bounded run on twenty coarse pieces caps how flat the cap can get. This is the layer-cake
statement of the limitation, and I can make it quantitative. My plateau is about fourteen of forty-one nodes,
so the top of the autoconvolution is roughly a third of its total support wide; to push `⟨t⟩/T` closer to the
box's `1/2` I would need the width `μ(t)` to stay near its base value further up toward `T` and then fall
off *sharply* — but a sharp fall-off is exactly a steep shoulder, and at twenty pieces the shoulder can only
descend in a few coarse steps (`v_14…v_19`, six heights) before it hits zero. The grid cannot render both a
wide plateau and a sharp shoulder at once; something has to be coarse. The `0.0044` gap to the published
twenty-step value is, in this light, just the small difference between my plateau-and-shoulder compromise and
theirs at the same resolution — both are pressed against the same forty-one-node ceiling, which is why
neither clears the low `0.89`s. If I am right that the search idea is sound and the resolution is the
ceiling, then the sharp falsifiable prediction is that *lifting this same shape onto a finer grid* — say five
times finer, giving on the order of two hundred nodes, room for both a much wider plateau and a much sharper
shoulder — should keep raising `R` past `0.885` toward the high `0.89`s, with no new idea about the shape
required, purely from letting `μ(t)` hold higher and fall faster. If instead a finer grid did *not* help,
that would falsify the resolution diagnosis and send me back to the shape itself. That is exactly the opening
for the next rung: take this optimized coarse profile, refine it onto a much finer grid, and let a
gradient-based method carve the fine structure that twenty heights simply cannot represent.
