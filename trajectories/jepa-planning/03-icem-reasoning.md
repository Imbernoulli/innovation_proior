CEM's numbers came back, and the first thing they told me is that the iteration-and-refit fix worked
exactly where I bet it would. The wandering collapsed: random's `mean_steps_to_success` of 64 / 82 / 112
fell to CEM's 24 / 23 / 46. Put ratios on that — `64/24 = 2.7`, `82/23 = 3.6`, `112/46 = 2.4` — so once
the Gaussian marches onto a coherent route through the door, successes arrive in roughly a third of the
steps random needed, and the biggest collapse is at horizon 60 where the refit had both enough horizon
to commit and few enough steps that the commitment mattered. The success rate rose at every horizon —
0.55 → 0.70 at 30, 0.70 → 0.85 at 60, 0.70 → 0.95 at 90 — and the largest single jump, `+0.25` at
horizon 90, is exactly where I predicted the floor's Brownian sequences fell shortest. And `mean_dist`
dropped hard, 12.6 / 9.6 / 8.2 down to 9.5 / 6.5 / 3.4. The refit is unambiguously the right move over
the floor.

But look closer at the *shape* across horizons, because that is where CEM is still leaving episodes on
the table, and the shape is the opposite of random's. At the long horizon (90) CEM is excellent — 0.95,
mean distance 3.4, the residual barely above the `4.5` success threshold. At the *short* horizon (30) it
is its weakest — only 0.70, mean distance 9.5, the highest residual of the three by a wide margin. That
inversion is the tell. Where random got *worse* with horizon (reach handicap growing like `√T`), CEM
gets *better* with horizon, which means CEM's bottleneck is not reach-per-plan — it is having enough
optimization budget, within a single `plan()` call, to both find a route and refine it. A short horizon
means each plan is only 30 steps, so the re-planning loop gets fewer corrections and each individual
optimization has to find a good route *the first time* inside one call — and CEM, drawing white Gaussian
perturbations and fitting a fresh distribution from a zero mean every call, spends most of its 20
iterations just discovering a route at all, with no budget left to refine it. The `0.70`-at-30 with a
`9.5` residual is CEM running out of optimization before it has committed to the door; at horizon 90 the
same 20 iterations have longer plans and many more re-plans to lean on, so the discovery-then-refine
sequence completes.

So two things are still bleeding, and they share a root. First, every `plan()` call starts cold: zero
mean, wide spread, rediscover the route from scratch — even though consecutive control steps face almost
the same planning problem, the goal hasn't moved, the route from one step earlier is still mostly valid.
CEM throws away the entire optimized distribution at the end of each step and the loop pays to re-find
it — the same amnesia I diagnosed in random search, one level up: random discarded scores within a call,
CEM discards the whole solution across calls. Second — and this bites the short horizon hardest — the
sampling is still white: CEM perturbs each timestep independently, so a sampled sequence is white noise
around the mean, and white actions integrate to a Brownian walk that barely ranges, so most perturbations
jitter near the current mean rather than proposing a committed, far-ranging excursion toward the door. At
a long horizon the re-planning loop papers over this; at a short horizon, where the optimizer must reach
the door inside one call, the white perturbations simply don't propose trajectories that get there, and
the elite fit has nothing good to lock onto. I want to fix the sample *shape* so each draw ranges farther
per unit of action energy, and stop throwing away the solution between steps and between iterations. The
budget is the enemy — 200 samples, 20 iters, each a full rollout — so every fix has to *reclaim* wasted
rollouts rather than ask for more.

Start with the sample shape, the bigger leak at the short horizon. The thing I actually want to control
is not the per-step variance — CEM already adapts that through the elite std — it is the *temporal
correlation* of the sampled sequence, which white noise has none of. How do I parameterize "amount of
temporal correlation" with one interpretable knob? Read the action sequence as a time series and look at
its power spectral density. White noise has a flat PSD: equal energy at all frequencies, including the
highest, which is the action flipping sign every timestep — pure diffusive jitter. To make the sequence
persist in a direction I want to *suppress* the high frequencies relative to the low ones, and the clean
one-knob family that does this is a power law `PSD_a(f) ∝ 1/f^β`, where `β = 0` recovers white noise and
larger `β` tilts energy toward the low frequencies — smoother, more persistent sequences. And it
propagates correctly to *position*: position is the integral of the action, integration is division by
`i·2πf` in the frequency domain, so `PSD_x(f) ∝ 1/f^{β+2}` — carrying the action spectrum through the
integrator reddens it by two more powers of `f`. Cranking `β` up piles more energy into the low
frequencies, i.e. larger, smoother, longer-range excursions at the *same* action variance, because `β`
only redistributes fixed total power from high frequencies (jitter) to low ones (drift). That is the
trade I want: same energy, more reach toward the door. The autocorrelation view says the same thing — a
power-law PSD has scale-free autocorrelation, no built-in correlation length capping how long the
sequence persists in a direction, which is exactly the property a far-ranging committed excursion has and
a Brownian walk (increments decorrelate after one step) does not.

I set `β = 2.5` — firmly in the smooth regime, since this task wants committed low-frequency motion
(cross the room, pass the door), not high-frequency control. The `1/f^{β+2}` reddening is a clean
identity only in the continuous-time limit; on a finite discrete horizon it is a *direction*, not an
exact exponent, but what I need is reach, and reach is what the low-frequency tilt delivers regardless of
the exact realized slope.

Drawing a length-T sequence with a `1/f^β` spectrum is cleanest in the frequency domain, because the PSD
*is* a frequency-domain statement: draw random complex Fourier coefficients, scale bin `k` by
`f_k^{−β/2}` (so `|coef|² ∝ 1/f^β`), inverse-FFT. I clamp the zero-frequency bin to the smallest
resolvable nonzero frequency `1/T` so `f^{−β/2}` does not blow up at DC. And there is a realness subtlety
that controls the *variance*, not just realness, so I cannot wave it away: for the inverse real-FFT to
produce a real sequence the DC coefficient must be real (and the Nyquist too, when T is even). A generic
complex bin carries variance in both parts — `E|sr + i·si|² = 2s²` — but forcing the imaginary part to
zero at DC leaves only `s²`, half the intended power at that bin. The `fmin` clamp sets DC's scale equal
to the first interior bin's, so their realized power ratio should be exactly 1; zeroing DC's imaginary
part and doing nothing else makes it `0.500`, and multiplying DC's surviving real part by `√2` — restoring
`Var(√2·sr) = 2s²` — makes it `1.001`. So `√2` is the exact compensation, and it earns its place not by
fixing the total sequence variance (DC and Nyquist are two of `~T/2` bins) but by holding the
*lowest-frequency* weight at full strength — and since large `β` piles most of the energy precisely at
DC, halving that bin would halve the very directional drift I am introducing colored noise *for*. The
colored sampler is otherwise a drop-in replacement for CEM's white `torch.randn`: same `[T, N, A]` shape,
scaled by the same elite std and added to the same mean, so the rest of the CEM machinery survives
unchanged and the only difference per sample is that it now ranges.

Now the amnesia leaks, at two timescales. *Within* a single call, across the inner iterations: each
iteration produces an elite set, and standard CEM uses it only to refit `μ, σ` and then discards the
actual sequences, drawing a fresh population from the shifted distribution — most of which will be worse
than the elites I just deleted. So I keep a fraction of them and add them back into the next iteration's
pool alongside the fresh colored draws. They are already scored, so reuse is nearly free — but only a
*fraction* (0.3), because the elites have small spread by construction, and if they dominate the pool the
refit collapses `σ` immediately and kills exploration before it starts. *Across* env steps: consecutive
control steps overlap by all but one timestep, so I shift the optimized mean forward by one, repeat the
last entry to fill the freed slot, and reset `σ` back to its wide initial value (the new final timestep is
genuinely unexplored, so it must re-open exploration). I also shift the kept elites — drop their executed
first timestep, append a freshly sampled final action — giving the new step a handful of genuinely-good
warm-start sequences, not just a shifted mean. This is the persistent state CEM lacked: I carry
`_mean, _std, _kept_elites` on the planner instance between calls, and reset them only when `t0=True`
flags an episode start (or when the horizon changes). The `t0` flag random and CEM both ignored is now
load-bearing — it is the one signal that tells me a warm start would be a *wrong* start, because the goal
and geometry just changed.

That across-step shift is worth its code because of how much it reclaims: the loop executes exactly one
action per call, so the new call's problem overlaps the old on `T−1` of its `T` timesteps — `29/30 = 97%`
at horizon 30, `99%` at horizon 90 — of the solution carries over unchanged. CEM was throwing away that
97–99% of a solution it had already paid 4000 rollouts to compute, and re-deriving it from a zero mean.
The warm start recovers almost all of it for free. And it explains why the payoff should be largest at
the *short* horizon even though the overlap fraction is larger at the long one: at horizon 30 the cold
restart cost CEM a proportionally huge share of its scarce 20 iterations on rediscovery, so returning
`29/30` of the solution frees exactly the budget the short horizon was starving for.

A few smaller reclaim-the-budget moves. CEM's `σ` shrinks automatically as iterations concentrate, so a
late iteration samples from an already-narrow distribution where each extra sample buys little — so I
decay the population geometrically, `N_i = max(2·elites, N/γ)` with `γ = 1.25`, floored at twice the
elite count so the elite selection stays meaningful. Starting from `N = 900` the ten iterations draw
`900, 720, …, 180`, summing to ~4101 fresh trajectories against CEM's flat `200 × 20 = 4000` — a
near-identical *total*, not bought compute, just reallocated: front-loaded so the early exploration-
critical iterations get the big populations and the late concentrated ones get the floor. That is exactly
what the short-horizon failure asked for. It also resolves the tension of running only ten iterations
rather than twenty — the failure was never a shortage of iterations per se, it was that each of CEM's 20
drew a small (200) white population from a cold start, so the exploration-critical early iterations were
the *most* starved; here the first iteration alone draws 900 colored sequences from a warm-started mean,
so route discovery happens densely in the first two or three warm iterations, leaving the decayed tail to
refine rather than discover. I also smooth the refit with momentum, `μ ← (1−α)·μ_elite + α·μ_prev` and
the same for `σ` with `α = 0.1`, because fitting a high-dimensional mean and std from a small elite set is
noisy and the momentum is free variance reduction. And I keep a *best-so-far* across all iterations: CEM
returns the final distribution mean, but that mean was never actually evaluated — it is the centroid of
the last elites, and in high dimensions a centroid can sit in a low-density region that rolls out worse
than the best sequence I actually scored. So I track the single lowest-cost evaluated sequence across
every iteration, roll the final mean through the model once at the end, and return whichever is actually
better. That stops the optimizer from discarding a known-good plan for an untested centroid, most valuable
at the short horizon where a single good route matters more than a well-fit distribution.

One honest note on what this task's harness does *not* let me use, relative to the full method this
descends from. That method is designed for a per-step MPC loop where the across-step shift is the headline
mechanism, leaning on a per-task `β` chosen for the control bandwidth and on executing the first action of
the *best evaluated* sequence each step. Here the loop already executes only the first action and
re-plans, the horizons are fixed by the benchmark rather than tuned, and I fold the best-evaluated idea
into the best-so-far + final-mean fallback rather than a per-iteration mean injection. So what I land is
the core — colored noise, elite reuse within and across steps, population decay, momentum, best-so-far —
fit to this task's single-plan-per-observation interface.

So the falsifiable expectations against CEM's numbers. The colored noise and warm-starting are aimed
straight at CEM's short-horizon weakness, so the cleanest prediction is at horizon 30: CEM's 0.70 with a
9.5 residual should rise, because committed colored excursions reach the door inside one call and the
across-step shift means each call no longer starts cold, with the residual finishing closer to the
threshold rather than running out of iterations short of it. At the long horizon, where CEM was already
0.95 with a 3.4 residual, I expect to roughly *hold* rather than leap — little headroom, and the
across-step machinery and best-so-far mainly protect against the centroid-in-a-bad-region failure rather
than adding reach. The risk I can already name is the mirror image of the reach I am buying: colored noise
plus aggressive elite reuse can *over-commit*. If the kept elites and the smooth low-frequency draws all
agree on a route that turns out wrong, the search has even less spread to escape than CEM did — the hard
top-k cut plus colored over-smoothing leaves no candidate that disagrees enough to pull the distribution
onto the other route, and where CEM's white sampling occasionally threw an outlier that reached the other
door, the colored draws are smooth and the reused elites concentrated, narrowing both escape hatches.
That is most dangerous exactly where CEM was strongest — the long horizon, where a committed wrong-side
route is a long one. If a long-horizon success rate slips below CEM's 0.95 even as the short horizon
improves, the diagnosis is written: the hard elite cut and the colored over-commitment are too greedy,
and the next move is a *soft*, cost-weighted update that lets a markedly-better rollout pull the
distribution without throwing away the spread a hard top-k discards. The full scaffold module for this
rung is in the answer.
