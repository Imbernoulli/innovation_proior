CEM's numbers came back, and the first thing they told me is that the iteration-and-refit fix worked
exactly where I bet it would. The wandering collapsed: random's `mean_steps_to_success` of 64 / 82 /
112 fell to CEM's 24 / 23 / 46, so once the Gaussian marches onto a coherent route through the door,
successes arrive in a quarter of the steps. The success rate rose at every horizon — 0.55 → 0.70 at
30, 0.70 → 0.85 at 60, 0.70 → 0.95 at 90 — and `mean_dist` dropped hard, 12.6 / 9.6 / 8.2 down to 9.5
/ 6.5 / 3.4. The refit is unambiguously the right move over the floor. But look closer at the *shape*
across horizons, because that is where CEM is still leaving episodes on the table. At the long horizon
(90) CEM is excellent — 0.95, mean distance 3.4, the elites converge on a clean long route. At the
*short* horizon (30) it is its weakest — only 0.70, mean distance 9.5, the highest residual distance
of the three. That is the tell. A short horizon means each plan is only 30 steps, so the re-planning
loop gets fewer corrections and each individual optimization has to find a good route *the first time*
inside one `plan()` call — and CEM, drawing white Gaussian perturbations and fitting a fresh
distribution from a zero mean every call, is spending most of its 20 iterations just discovering a
route at all, then has no budget left to refine it. The 0.70-at-30 with a 9.5 residual is CEM running
out of optimization before it has committed to the door.

So two things are still bleeding, and I think they share a root. First, every `plan()` call starts
cold: zero mean, wide spread, rediscover the route from scratch — even though consecutive control
steps face almost the same planning problem, the goal hasn't moved, the route from one step earlier is
still mostly valid. CEM throws away the entire optimized distribution at the end of each step and the
loop pays to re-find it. That is the same amnesia I diagnosed in random search, just one level up: random
discarded scores *within* a call, CEM discards the whole solution *across* calls. Second — and this is
what bites the short horizon hardest — the white-noise sampling. CEM perturbs each timestep
independently, so a sampled action sequence is still white noise around the mean, and white actions
integrate to a Brownian walk: net displacement grows only like the square root of the horizon, so most
sampled perturbations jitter near the current mean rather than proposing a committed, far-ranging
excursion toward the door. At a long horizon the re-planning loop papers over this; at a short horizon,
where the optimizer has to reach the door inside one call, the white perturbations simply don't propose
trajectories that get there, and the elite fit has nothing good to lock onto. I want to fix the sample
*shape* so each draw ranges farther per unit of action energy, and I want to stop throwing away the
solution between steps and between iterations. The whole budget is the enemy — 200 samples, 20 iters,
each sample a full world-model rollout — so every fix has to *reclaim* wasted rollouts rather than ask
for more.

Start with the sample shape, because I think it is the bigger leak at the short horizon. What does a
single sampled action sequence do when I roll it through the model? It drives the latent state
somewhere, and to first order position is the running integral of the action. If the action sequence
is white — independent per-timestep increments — then position is a Brownian walk, and a Brownian walk
famously does not go anywhere: independent increments cancel, expected squared displacement grows only
linearly in horizon, so a fixed budget of action energy buys only a small net excursion. That is
exactly the wrong behavior in a maze where the goal sits on the far side of a wall: I need sampled
trajectories that *commit to a direction* long enough to reach and pass the door, and white noise
structurally won't. The thing I actually want to control is not the per-step variance — CEM already
adapts that through the elite std — it is the *temporal correlation* of the sampled sequence. White
noise has none.

How do I parameterize "amount of temporal correlation" with one interpretable knob? Read the action
sequence as a time series and look at its power spectral density — how much energy sits at each
temporal frequency. White noise has a flat PSD: equal energy at all frequencies, including the very
highest, which corresponds to the action flipping sign every timestep — pure diffusive jitter. To make
the sequence persist in a direction I want to *suppress* the high frequencies relative to the low ones.
The clean one-knob family that does this is a power-law spectrum, PSD(f) ∝ 1/f^β, where β = 0 recovers
white noise and larger β tilts energy toward the low frequencies — smoother, more persistent
sequences. One scalar dials from white to very smooth. And I can check this propagates correctly to
*position*, not just the action: carrying a 1/f^β action spectrum through the integrator reddens it by
two more powers of f, so position has spectrum 1/f^(β+2) — even white actions give the 1/f² Brownian
walk (the sanity check), and cranking β up piles even more energy into the low frequencies, i.e. larger,
smoother, longer-range excursions at the *same* action variance, because β only redistributes energy
across frequencies, it never adds any. That is the trade I want: same energy, more reach toward the
door. I set β = 2.5 — firmly in the smooth regime, since this task wants committed low-frequency
motion (cross the room, pass the door), not high-frequency control.

Drawing a length-T sequence with a 1/f^β spectrum is cleanest in the frequency domain, because the PSD
*is* a frequency-domain statement: draw random complex Fourier coefficients, scale bin k by f_k^(-β/2),
inverse-FFT. I have to handle the zero-frequency bin — f^(-β/2) blows up at f = 0 — by clamping the
lowest frequency to the smallest resolvable nonzero frequency, 1/T, so DC is finite. And there is a
realness subtlety that controls the *variance*, so I cannot wave it away: for the inverse real-FFT to
produce a real sequence, the DC coefficient must be real (and the Nyquist coefficient too, when T is
even). Zeroing those imaginary parts deletes half the variance at those bins, so I compensate by
multiplying the surviving real part there by √2 — restoring the full intended power. Without the √2 the
DC drift comes out weaker than designed, which is precisely the long-range directional component I am
introducing colored noise *for*. I keep the √2. The colored sampler is then a drop-in replacement for
CEM's white `torch.randn`: same interface, returns the same `[T, N, A]` shape, scaled by the same elite
std and added to the same mean — so the entire rest of the CEM machinery survives unchanged, and the
only thing different per sample is that it now ranges.

Now the amnesia leaks, and there are two at two timescales. *Within* a single `plan()` call, across
the inner iterations: each iteration produces an elite set — the lowest-cost sequences found so far —
and standard CEM uses them only to refit μ, σ and then discards the actual sequences, drawing a fresh
population from the shifted distribution. But those elites were *good*; after the small refit most of
the new population will be worse than the elites I just deleted. So I keep a fraction of them and add
them back into the next iteration's candidate pool, alongside the fresh colored draws. They are even
cheaper than fresh samples in a sense — but I must add only a *fraction* (0.3), because the elites have
small spread by construction, and if they dominate the pool the refit collapses σ immediately and kills
exploration before it has started. *Across* env steps: consecutive control steps overlap by all but one
timestep, so I shift the optimized mean forward by one, repeat the last entry to fill the freed slot,
and reset σ back to its initial wide value (the new final timestep is genuinely unexplored, so it must
re-open exploration). I also shift the kept elites: drop their executed first timestep and append a
freshly sampled final action — giving the new step a handful of genuinely-good warm-start sequences, not
just a shifted mean. This is the persistent state CEM lacked: I carry `_mean`, `_std`, `_kept_elites`
on the planner instance between calls, and reset them only when `t0=True` flags an episode start (or
when the horizon changes). The `t0` flag that random and CEM both ignored is now load-bearing.

A few smaller reclaim-the-budget moves, each closing a specific leak. CEM's σ shrinks automatically as
iterations concentrate, so a late iteration samples from an already-narrow distribution where each
extra sample buys little — so I decay the population geometrically, N_i = max(2·elites, N/γ) with γ =
1.25, floored at twice the elite count so the elite selection and refit stay meaningful. The payoff is
direct: cheaper late iterations mean the *same* total rollout budget buys more refinement than fixed-N
CEM could afford — exactly what I need against the short-horizon failure where CEM ran out of
optimization. I smooth the refit with momentum, μ ← (1−α)·μ_elite + α·μ_prev and the same for σ, with
α = 0.1, because fitting a high-dimensional mean and std from a small elite set is noisy and the
momentum is free variance reduction. And I keep a *best-so-far* across all iterations: CEM returns the
final distribution mean, but that mean was never actually evaluated — it is the centroid of the last
elites and in high dimensions a centroid can sit in a low-density region that rolls out worse than the
best sequence I actually scored. So I track the single lowest-cost evaluated sequence across every
iteration, and at the end I roll the final mean through the model once and return whichever is actually
better — the best-ever evaluated trajectory or the final mean. That stops the optimizer from
discarding a known-good plan in favor of an untested centroid, which is most valuable precisely at the
short horizon where a single good route matters more than a well-fit distribution.

One honest note on what this task's harness does *not* let me use, relative to the full method this
descends from. That method is designed for a per-step MPC loop where the across-step shift is the
headline mechanism, and it leans on a per-task β chosen for the control bandwidth and on executing the
first action of the *best evaluated* sequence each step. Here the loop already executes only the first
action and re-plans, the horizons are fixed by the benchmark rather than tuned, and I fold the
best-evaluated idea into the best-so-far + final-mean fallback rather than a per-iteration mean
injection. So what I am landing is the core: colored noise, elite reuse within and across steps,
population decay, momentum, best-so-far — fit to this task's single-plan-per-observation interface, not
the full MPC harness it was designed for. I also bump the budget to match the regime this method was characterized
in: 900 samples, 10 iterations, K = 10%, which after the γ = 1.25 decay is a comparable total rollout
count to CEM's 200×20 but spent with the colored, reuse-and-decay structure instead of flat white
re-sampling.

So the falsifiable expectations against CEM's numbers. The colored noise and warm-starting are aimed
straight at CEM's short-horizon weakness, so the cleanest prediction is at horizon 30: CEM's 0.70 with
a 9.5 residual should rise, because committed colored excursions reach the door inside one call and the
across-step shift means each call no longer starts cold. I expect the short-horizon `mean_dist` to drop
the most — from 9.5 toward the 4–6 range — because even on failures the colored planner should finish
closer. At the long horizon, where CEM was already 0.95, I expect to roughly *hold* rather than leap —
there is little headroom, and the across-step machinery and best-so-far fallback mainly protect against
the centroid-in-a-bad-region failure. The risk I can already name: colored noise plus aggressive elite
reuse can over-commit — if the kept elites and the smooth low-frequency draws all agree on a route that
turns out wrong, the search has even less spread to escape than CEM did, so a long-horizon seed could
*regress* if the committed route is the wrong side of the door. If that shows up — a long-horizon
success rate that slips below CEM's 0.95 even as the short horizon improves — the diagnosis is already
written: the hard elite cut and the colored over-commitment are too greedy, and the next move is a
*soft*, cost-weighted update that lets a markedly-better rollout pull the distribution without throwing
away the spread a hard top-k discards. The full scaffold module — the literal `CustomPlanner` fill for
this rung — is in the answer.
