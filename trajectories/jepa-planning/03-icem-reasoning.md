CEM's numbers came back, and the first thing they told me is that the iteration-and-refit fix worked
exactly where I bet it would. The wandering collapsed: random's `mean_steps_to_success` of 64 / 82 /
112 fell to CEM's 24 / 23 / 46. Put ratios on that — `64/24 = 2.7`, `82/23 = 3.6`, `112/46 = 2.4` — so
once the Gaussian marches onto a coherent route through the door, successes arrive in roughly a third
of the steps random needed, and the biggest collapse is at horizon 60 where the refit had both enough
horizon to commit and few enough steps that the commitment mattered. The success rate rose at every
horizon — 0.55 → 0.70 at 30, 0.70 → 0.85 at 60, 0.70 → 0.95 at 90 — and the largest single jump, `+0.25`
at horizon 90, is exactly where I predicted the floor's Brownian sequences fell shortest, so the refit
buys the most where reach was scarcest. And `mean_dist` dropped hard, 12.6 / 9.6 / 8.2 down to 9.5 /
6.5 / 3.4. The refit is unambiguously the right move over the floor.

But look closer at the *shape* across horizons, because that is where CEM is still leaving episodes on
the table, and the shape is the opposite of random's. At the long horizon (90) CEM is excellent — 0.95,
mean distance 3.4, the elites converge on a clean long route and the residual is barely above the `4.5`
success threshold. At the *short* horizon (30) it is its weakest — only 0.70, mean distance 9.5, the
highest residual of the three by a wide margin (`9.5` versus `6.5` and `3.4`). That inversion is the
tell. Where random got *worse* with horizon (reach handicap growing like `√T`), CEM gets *better* with
horizon, which means CEM's bottleneck is not reach-per-plan — it is having enough optimization budget,
within a single `plan()` call, to both find a route and refine it. A short horizon means each plan is
only 30 steps, so the re-planning loop gets fewer corrections and each individual optimization has to
find a good route *the first time* inside one call — and CEM, drawing white Gaussian perturbations and
fitting a fresh distribution from a zero mean every call, is spending most of its 20 iterations just
discovering a route at all, then has no budget left to refine it. The `0.70`-at-30 with a `9.5` residual
is CEM running out of optimization before it has committed to the door. At horizon 90 the same 20
iterations have longer plans and many more re-plans to lean on, so the discovery-then-refine sequence
completes; at horizon 30 it does not.

So two things are still bleeding, and I think they share a root. First, every `plan()` call starts
cold: zero mean, wide spread, rediscover the route from scratch — even though consecutive control steps
face almost the same planning problem, the goal hasn't moved, the route from one step earlier is still
mostly valid. CEM throws away the entire optimized distribution at the end of each step and the loop
pays to re-find it. That is the same amnesia I diagnosed in random search, just one level up: random
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
sequence as a time series and look at its power spectral density — how much energy sits at each temporal
frequency. White noise has a flat PSD: equal energy at all frequencies, including the very highest,
which corresponds to the action flipping sign every timestep — pure diffusive jitter. To make the
sequence persist in a direction I want to *suppress* the high frequencies relative to the low ones. The
clean one-knob family that does this is a power-law spectrum, `PSD_a(f) ∝ 1/f^β`, where `β = 0` recovers
white noise and larger `β` tilts energy toward the low frequencies — smoother, more persistent
sequences. One scalar dials from white to very smooth. And I can check this propagates correctly to
*position*, not just the action. Position is the integral of the action; in the frequency domain
integration is division by `i·2πf`, so `PSD_x(f) = PSD_a(f) / (2πf)² ∝ 1/f^{β+2}` — carrying a `1/f^β`
action spectrum through the integrator reddens it by two more powers of `f`. Two consistency checks
fall out. First, even white actions (`β = 0`) give position spectrum `1/f²`, which is precisely the
Brownian walk — the sanity check that the machinery reproduces the failure I already understand.
Second, cranking `β` up piles even more energy into the low frequencies, i.e. larger, smoother,
longer-range excursions at the *same* action variance, because `β` only redistributes energy across
frequencies, it never adds any — the total power is fixed by the sampling std, and `β` only moves it
from high frequencies (jitter) to low ones (drift). That is the trade I want: same energy, more reach
toward the door.

I can say a little more about *why* correlation buys reach by looking at the autocorrelation instead of
the spectrum, because that is where the "no built-in length scale" intuition lives. The autocorrelation
is the inverse transform of the PSD, and for a power-law PSD it is self-similar: rescale time `τ → sτ`
and the autocorrelation rescales as `s^{β−1}` times itself, with no additive constant that would set a
correlation length. Scale-free autocorrelation means there is no built-in cutoff that caps how long the
sequence persists in a direction — which is exactly the property a far-ranging, committed excursion has
and a Brownian walk (whose increments decorrelate after one step) does not. So a redder action sequence
pushes consistently in some direction over a stretch of the horizon rather than reversing every step,
and that is the mechanical reason it covers more ground per unit of energy. I set `β = 2.5` — firmly in
the smooth regime, since this task wants committed low-frequency motion (cross the room, pass the door),
not high-frequency control. The `1/f^{β+2}` reddening is a clean identity only in the continuous-time
limit; on a finite discrete horizon it is a *direction*, not an exact exponent, but what I actually need
is reach, and reach is what the low-frequency tilt delivers regardless of the exact realized slope.

Drawing a length-T sequence with a `1/f^β` spectrum is cleanest in the frequency domain, because the
PSD *is* a frequency-domain statement: draw random complex Fourier coefficients, scale bin `k` by
`f_k^{−β/2}` (so `|coef|² ∝ 1/f^β`), inverse-FFT. I have to handle the zero-frequency bin — `f^{−β/2}`
blows up at `f = 0` — by clamping the lowest frequency to the smallest resolvable nonzero frequency,
`1/T`, so DC is finite. And there is a realness subtlety that controls the *variance*, so I cannot wave
it away: for the inverse real-FFT to produce a real sequence, the DC coefficient must be real (and the
Nyquist coefficient too, when T is even). A generic complex bin carries variance in both its real and
imaginary parts — `E|sr + i·si|² = 2·s²` if each part has scale `s` — but if I force the imaginary part
to zero at DC I am left with only `s²`, half the intended power at that bin. Let me verify the fix is
the right size rather than guess it. The `fmin` clamp sets DC's scale equal to the first interior bin's
scale, so DC and bin 1 have *identical* intended power and their realized power ratio should be exactly
1; if I zero DC's imaginary part and do nothing else, that ratio comes out `0.500` (half the power),
and if I multiply DC's surviving real part by `√2` — restoring `Var(√2·sr) = 2s²` — the ratio comes out
`1.001`. So `√2` is exactly the compensation, not approximately: it earns its place not by fixing the
total sequence variance (DC and Nyquist are only two of `~T/2` bins, so the total barely moves) but by
holding the *lowest-frequency* weight at full strength — and since large `β` piles most of the energy
precisely at DC, halving that bin would be halving the very directional drift I am introducing colored
noise *for*. I keep the `√2`. The colored sampler is then a drop-in replacement for CEM's white
`torch.randn`: same interface, returns the same `[T, N, A]` shape, scaled by the same elite std and
added to the same mean — so the entire rest of the CEM machinery survives unchanged, and the only thing
different per sample is that it now ranges.

Now the amnesia leaks, and there are two at two timescales. *Within* a single `plan()` call, across the
inner iterations: each iteration produces an elite set — the lowest-cost sequences found so far — and
standard CEM uses them only to refit `μ, σ` and then discards the actual sequences, drawing a fresh
population from the shifted distribution. But those elites were *good*; after the small refit most of
the new population will be worse than the elites I just deleted. So I keep a fraction of them and add
them back into the next iteration's candidate pool, alongside the fresh colored draws. They are already
scored, so reusing them is nearly free — but I must add only a *fraction* (0.3), because the elites have
small spread by construction, and if they dominate the pool the refit collapses `σ` immediately and
kills exploration before it has started. With `elites_size = 90` (of which more below), that is `int(90
· 0.3) = 27` reused sequences per iteration — a minority of a several-hundred-sequence pool, enough to
retain the good ones without swamping the fresh exploration. *Across* env steps: consecutive control
steps overlap by all but one timestep, so I shift the optimized mean forward by one, repeat the last
entry to fill the freed slot, and reset `σ` back to its initial wide value (the new final timestep is
genuinely unexplored, so it must re-open exploration). I also shift the kept elites: drop their executed
first timestep and append a freshly sampled final action — giving the new step a handful of
genuinely-good warm-start sequences, not just a shifted mean. This is the persistent state CEM lacked: I
carry `_mean`, `_std`, `_kept_elites` on the planner instance between calls, and reset them only when
`t0=True` flags an episode start (or when the horizon changes). The `t0` flag that random and CEM both
ignored is now load-bearing — it is the one signal that tells me a warm start would be a *wrong* start,
because the goal and geometry just changed.

Let me quantify how much the across-step shift actually reclaims, because it decides whether this leak
is worth the code. The loop executes exactly one action per `plan()` call, then re-plans a fresh T-step
horizon. So the new call's planning problem overlaps the old one on `T−1` of its `T` timesteps: the old
plan's steps 2…T are still the right thing to do for the new plan's steps 1…T−1, only the single new
final timestep is genuinely unseen. That is `(T−1)/T` of the horizon — `29/30 = 97%` at horizon 30,
`89/90 = 99%` at horizon 90 — of the optimized solution that carries over unchanged. CEM was throwing
away `97–99%` of a solution it had already paid `4000` rollouts to compute, and re-deriving it from a
zero mean on the next call. The warm start recovers almost all of it for free; the shift is not a
marginal trick, it is refusing to re-solve a problem that is `97%` identical to the one just solved. And
it explains why the payoff should be largest at the *short* horizon even though the overlap fraction is
larger at the long one: at horizon 30 the cold restart cost CEM a proportionally huge share of its
scarce 20 iterations on rediscovery, so returning `29/30` of the solution frees exactly the budget the
short horizon was starving for.

A few smaller reclaim-the-budget moves, each closing a specific leak. CEM's `σ` shrinks automatically as
iterations concentrate, so a late iteration samples from an already-narrow distribution where each extra
sample buys little — so I decay the population geometrically, `N_i = max(2·elites, N/γ)` with `γ = 1.25`,
floored at twice the elite count so the elite selection and refit stay meaningful. Let me actually add
up what that decay spends against CEM's flat budget, because the whole point is to spend the same total
differently. Starting from `N = 900` and flooring at `2·90 = 180`, the ten iterations draw `900, 720,
576, 460, 368, 294, 235, 188, 180, 180` fresh sequences, which sum to `4101`. CEM's flat budget is
`200 × 20 = 4000`. So the decayed schedule is a near-identical *total* rollout count — I am not buying
compute, I am reallocating it — but it is front-loaded: the early iterations, where exploration matters
and the distribution is wide, get the big populations (`900`, `720`), and the late iterations, where the
distribution has already concentrated and each sample is redundant, get the floor (`180`). That
reallocation is exactly what the short-horizon failure asked for: more effective exploration up front to
*discover* the route, without wasting the tail on a distribution that has already committed. I smooth
the refit with momentum, `μ ← (1−α)·μ_elite + α·μ_prev` and the same for `σ`, with `α = 0.1`, because
fitting a high-dimensional mean and std from a small elite set is noisy and the momentum is free
variance reduction. And I keep a *best-so-far* across all iterations: CEM returns the final distribution
mean, but that mean was never actually evaluated — it is the centroid of the last elites, and in high
dimensions a centroid can sit in a low-density region that rolls out worse than the best sequence I
actually scored. So I track the single lowest-cost evaluated sequence across every iteration, and at the
end I roll the final mean through the model once and return whichever is actually better — the best-ever
evaluated trajectory or the final mean. That stops the optimizer from discarding a known-good plan in
favor of an untested centroid, which is most valuable precisely at the short horizon where a single good
route matters more than a well-fit distribution.

The budget numbers above already stated the override, but let me say why I take it. The harness passes
CEM/MPPI defaults of `(200, 20)`, and I override to `(900, 10)` with `elites_size = num_samples//10 =
90` and the `γ = 1.25` decay, so the total after decay (`~4101`) is comparable to CEM's `4000` but spent
with the colored, reuse-and-decay structure instead of flat white re-sampling. Ten iterations rather
than twenty is affordable precisely because the reuse and warm-start mean each iteration starts from a
better place, so fewer are needed; and the larger per-iteration population is what makes colored
exploration effective before the decay kicks in. There is a coherence check worth doing here: CEM's
short-horizon failure was running out of optimization *before* discovering a route, so does spending the
same total budget over fewer, front-loaded iterations actually help that? It does, because the failure
was never a shortage of iterations per se — it was that each of CEM's 20 iterations drew a small (200)
white population from a cold start, so the early, exploration-critical iterations were the *most*
starved. Here the first iteration alone draws 900 colored sequences from a warm-started mean, so the
route discovery that CEM spread thinly across many cold iterations happens densely in the first two or
three warm ones, leaving the decayed tail to refine rather than discover. The reallocation moves budget
to exactly the phase the short horizon was starving.

One honest note on what this task's harness does *not* let me use, relative to the full method this
descends from. That method is designed for a per-step MPC loop where the across-step shift is the
headline mechanism, and it leans on a per-task `β` chosen for the control bandwidth and on executing the
first action of the *best evaluated* sequence each step. Here the loop already executes only the first
action and re-plans, the horizons are fixed by the benchmark rather than tuned, and I fold the
best-evaluated idea into the best-so-far + final-mean fallback rather than a per-iteration mean
injection. So what I am landing is the core: colored noise, elite reuse within and across steps,
population decay, momentum, best-so-far — fit to this task's single-plan-per-observation interface, not
the full MPC harness it was designed for.

So the falsifiable expectations against CEM's numbers. The colored noise and warm-starting are aimed
straight at CEM's short-horizon weakness, so the cleanest prediction is at horizon 30: CEM's 0.70 with
a 9.5 residual should rise, because committed colored excursions reach the door inside one call and the
across-step shift means each call no longer starts cold. I expect the short-horizon `mean_dist` to drop
the most — from 9.5 toward the 4–6 range — because even on failures the colored planner should finish
closer, having centered on the door rather than run out of iterations short of it. At the long horizon,
where CEM was already 0.95 with a 3.4 residual, I expect to roughly *hold* rather than leap — there is
little headroom, and the across-step machinery and best-so-far fallback mainly protect against the
centroid-in-a-bad-region failure rather than adding reach that horizon already has. The risk I can
already name, and it is the mirror image of the reach I am buying: colored noise plus aggressive elite
reuse can over-commit. If the kept elites and the smooth low-frequency draws all agree on a route that
turns out wrong, the search has even *less* spread to escape than CEM did — the hard top-k elite cut
plus the colored over-smoothing leaves no candidate that disagrees enough to pull the distribution onto
the other route. Where CEM's anisotropic collapse could at least occasionally be rescued by a white
outlier that happened to reach the other door, the colored draws are smooth and the reused elites are
concentrated, so both of CEM's escape hatches are narrowed. This is most dangerous exactly where CEM was
strongest — the long horizon, where a committed wrong-side route is a long committed wrong-side route.
If that shows up — a long-horizon success rate that slips below CEM's 0.95 even as the short horizon
improves — the diagnosis is already written: the hard elite cut and the colored over-commitment are too
greedy, and the next move is a *soft*, cost-weighted update that lets a markedly-better rollout pull the
distribution without throwing away the spread a hard top-k discards. The full scaffold module — the
literal `CustomPlanner` fill for this rung — is in the answer.
