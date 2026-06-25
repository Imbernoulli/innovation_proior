The flat profile sat exactly at `2/3`, and I want to be sure I understand why before I move. With every
piece identical the autoconvolution of a step function is the discrete triangle: convolving `[1,1,1]` with
itself gives node values `1,2,3,2,1`, a tent. For the continuous limit I can do the ratio by hand. A
triangle of height `h` on `[-1,1]`, peaking at `0`, has `||·||_inf = h`, `||·||_1 = h` (area of a base-`2`
height-`h` triangle), and `||·||_2^2 = ∫_{-1}^{1} h^2(1-|x|)^2 dx = 2h^2∫_0^1(1-x)^2 dx = 2h^2/3`. So
`R = (2h^2/3)/(h·h) = 2/3`, independent of `h`. That matches the step computation to six digits, and it makes
the obstruction concrete: with the heights flat the autoconvolution is locked to a tent and there is no
gradient to follow — refining the grid does nothing, because `R` sees only the *shape* of the tent, not its
resolution. (I checked: flat profiles on `3`, `20`, and `100` pieces all return `0.666667`.) So the only way
forward is to introduce *variation* among the heights and let some search find a non-flat profile that bends
the autoconvolution away from the tent. The questions are how to search, and at what scale.

Scale first, because it decides everything else. I am tempted to go straight to a large piece count, but
that is a trap at this stage. The functional is highly non-convex — many local optima, lots of symmetry to
break — and a long height vector is a high-dimensional search space where a blind local search wanders for
a very long time before finding anything good. The smart move is to find the *shape* at low resolution
first, where the vector is short enough to explore thoroughly, and only later lift that shape to finer
grids. The literature points exactly here: Matolcsi and Vinuesa got past `0.88` with only `20` steps, and
later constructions seeded their large-`N` profiles from a small optimized one of this kind. So I will work
at `N` around `20`: short enough that a stochastic search can canvas the shape space, long enough that the
autoconvolution has real internal structure to exploit.

Now the search itself. My first instinct is the cheapest thing that could work — greedy hill-climbing: from
some start, repeatedly try nudging each height up or down and take the best improving move, stopping when
none improves. Before committing to anything fancier I should actually see how far that gets, because if
greedy already reaches the high `0.88`s I don't need annealing at all. So I ran it. Starting from the flat
profile it climbs — to `0.7963`, where it sticks at a profile that has settled into a two-level block,
roughly half the pieces at one height and half at double — and starting from four different random seeds it
stalls at `0.756`, `0.800`, `0.810`, `0.768`. Every run dies in the `0.75`–`0.81` band, well short of the
`0.88` I am after.

That result also corrects an assumption I almost wrote down. I had expected the flat profile itself to be
the trap — that any small perturbation would *lower* `R` and a greedy climber would simply never leave
`2/3`. But when I perturb a single height of the flat profile and measure, the moves come out roughly fifty
-fifty: of `2000` single-coordinate kicks, `986` raised `R` and `1014` lowered it. So the flat point is not
a strict local maximum at all; greedy leaves it immediately and climbs into the high `0.7`s. The real
obstruction is one rung up: the landscape is full of *moderate* local optima — two-level blocks and the
like — in the `0.75`–`0.81` range, and greedy parks at the first one it meets. Crossing out of *those*
basins, not out of the flat point, is the thing I need a smarter search for.

That is what changes my mind toward simulated annealing. If the barriers are between mediocre local optima
and better ones, I have to be willing to accept moves that make the ratio temporarily *worse* and walk
downhill across a ridge before climbing the far side. So: propose a perturbation to one height; if it
improves `R`, take it; if it worsens `R`, take it anyway with a Metropolis probability `exp((R'-R)/T)` set
by a temperature `T` that I cool over the run. Hot early, so the search wanders freely and is not captured
by the first `0.78` basin it falls into; cold late, so it settles into whatever better basin it has found.
The bet is that with enough wandering it discovers a profile whose autoconvolution has a flat cap and steep
sides, above the `0.81` ceiling greedy hit.

I should check the temperature is in a sane range rather than guess, because annealing is only useful if the
acceptance probabilities are neither always-`1` nor always-`0` across the schedule. Near a good profile, a
single multiplicative kick changes `R` by a median of about `1.3e-3` (`90`th percentile `~8e-3`). At the
hot end `T ≈ 2e-3`, a median worsening is accepted with probability `exp(-1.3e-3/2e-3) ≈ 0.52` — the search
genuinely wanders. At the cold floor `T ≈ 1e-6`, the same worsening is accepted with probability
`exp(-1.3e-3/1e-6) ≈ 0` — the search has frozen into pure hill-climbing. So a geometric cool from `~3e-3`
down to `~1e-6` does exactly what I want it to: free exploration early, fine settling late. The objective is
already bounded in `[0,1]` and its single-perturbation changes are small and well-scaled, so unlike some
problems I do not need to take a log or rescale; I can anneal directly on `R`.

A few design decisions need care, and they come from the geometry of this objective. The first is the
perturbation. Heights are non-negative and likely span a wide dynamic range — I expect a tall spike or two
and a long shoulder of smaller values, not a uniform spread. A single additive Gaussian kick of fixed size
would be far too coarse for the small heights and far too timid for the large ones. So I make the kick
*multiplicative in scale*: perturb a randomly chosen height by an amount proportional to its own magnitude
plus a small floor, and reflect any negative result back to be non-negative so the candidate stays legal.
This adjusts a tall spike and a thin shoulder value on comparable *relative* terms, which is the right
invariance for a scale-free objective. I shrink the perturbation scale alongside the temperature so that
late in the run the search makes fine adjustments to a settled shape rather than large jumps.

The second is restarts and seeding. A single annealing run can still get trapped, so I run several
independent restarts — some from a smooth single-bump seed (a Gaussian-shaped profile, which I expect to be
in the right neighborhood because the good autoconvolutions are unimodal), some from pure random heights —
and keep the best profile any restart ever reaches. The single-bump seed is the educated guess: if the
autoconvolution wants a flat cap, the height profile producing it is likely concentrated, so starting
concentrated should land in the good basin faster.

The third is doing the local refinement well, because annealing alone is a blunt instrument for the last
digits. Between and after the stochastic phases I would like to take real gradient steps, but the objective
has a non-differentiable `max` in the denominator (`||f*f||_inf`). I can smooth it: replace `max_j L_j` with
the softmax `(1/β) log Σ_j exp(β L_j)`, which is differentiable and overestimates the true max — so the
surrogate *underestimates* `R` and tightens to it as `β → ∞`. I checked the surrogate on a candidate whose
true `R` is `0.8848`: at `β=5` it reads `0.5642`, at `β=40` `0.8297`, at `β=400` `0.8792`, at `β=6000`
`0.8844` — a clean monotone climb toward the exact value. So a ladder of increasing `β` fed to L-BFGS
should track the real objective ever more faithfully, and that is what I use to do the gradient legwork.

I want to know how much of the `0.88` band the gradient ladder reaches on its own, before any annealing
polish, so I ran a multistart of the `β`-ladder from `30` mixed seeds. The best of those reached `0.8790`
— already across the `0.81` ceiling greedy could not pass, and within a whisker of the published `20`-step
numbers. Layering the exact-ratio annealing polish on top of that best profile is what carries the last
fraction of a percent.

When the dust settles, the converged `20`-height profile does something I flagged as a thing to watch: it
drives heights to *zero*. Six of the `20` are exactly `0` (a seventh is `1.2e-4`), and the nonzero support
is not contiguous — it is a sharp spike near the front, a short gap, then a shaped shoulder. So the best
`20`-piece profile does not use all `20` pieces; it wants a specific *sparse, asymmetric* support. And its
autoconvolution earns its score honestly: the node values rise to a peak of `1.0` and then hold there — `17`
of the `41` nodes sit within `10%` of the peak — a genuine flat cap with steep sides, exactly the tent
-breaking shape I was hoping variation would buy. The final `R` is `0.884823`.

So this rung lands where the principle predicts. Breaking the flat symmetry and annealing the heights buys a
large jump, from `0.6667` into the high `0.88`s, the band where the known `20`-step constructions live. I do
not beat the published `0.88922` by a meaningful margin, and I did not expect to: that number is itself the
product of careful optimization, and `20` pieces is a coarse grid that caps how flat the autoconvolution's
cap can get. The thing capping me here is resolution, not the search idea — `20` pieces cannot render an
arbitrarily flat-topped autoconvolution. That stall is the opening for the next rung: lift this optimized
coarse shape onto a much finer grid and let gradient-based refinement carve the fine structure that
annealing on `20` heights cannot represent.
