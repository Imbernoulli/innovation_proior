The quantile head did the thing I bet on, and the LunarLander column proves it. The worst seed climbed
clean out of the crash basin: where the dueling head had {127.35, 229.07, **−89.25**} for a mean of 89,
QR-DQN posted {152.64, 243.81, **194.99**} for a mean of 197.15 — the negative seed is gone, the spread
collapsed from a 318-point range to a 91-point range, and every seed is now solidly positive. That is
exactly the prediction: modeling the return distribution let the greedy policy stop being fooled by a
high mean that hid crash mass. CartPole pinned at a clean 500 on all three seeds — the 50-way head did
not destabilize the already-solved task, which was the risk I flagged. Acrobot landed at −80.07, a hair
better than dueling's −82.6 and right in the low −80s as expected, since its return is nearly unimodal
and there is little distributional structure for quantiles to exploit. So the change worked. But I closed
the last step with a specific doubt, and the LunarLander mean of 197 is exactly where that doubt lives:
197 is good, not great, and the per-seed numbers are still ragged — seed 42 at 152 is well below seed 123
at 244. A consistently-landing LunarLander agent should be scoring in the 200s on every seed. The
quantile picture of the return is still coarse where it matters most.

Let me quantify "ragged," because vague dissatisfaction is not a design brief. The three LunarLander seeds
are {152.64, 243.81, 194.99}: a range of 91.17 and a seed-to-seed standard deviation of about 37 around
the 197.15 mean. That is a 3.5× tightening of the dueling range (318.32 → 91.17) — real progress, the
distributional representation earned its keep — but 37 is still a wide band on a task where a competent
lander should clear 200 on every seed. Seed 42 at 152.64 sits 44.5 below the mean; seed 123 at 243.81
sits 46.7 above; the gap between those two is essentially the entire reason the mean is 197 and not ~240.
Nothing is catastrophic now — no seed is in the negative basin — but the ceiling is being pinned by
whichever seeds land *marginally*, and a marginal LunarLander landing is exactly the regime where a
slightly mis-estimated crash probability tips a close call the wrong way. So the residual is neither a
mean problem nor a crash-basin problem; it is a *tail-resolution* problem — the estimate of how much
probability sits out in the large-negative region is noisy, and that noise leaks into the mean the greedy
policy ranks on. That diagnosis is specific enough to design against.

Let me name the doubt precisely, because it dictates the next move. QR-DQN learns `N = 50` equal-mass
locations — quantiles — and lets them slide to wherever the returns live. The strength of that is the
support adapts and there are no bounds to set. The weakness is exactly the dual: uniform mass means the
*resolution* is spread evenly across the probability axis, so the low-probability crash tail of
LunarLander — the rare, large-negative outcome that decides the worst seeds — gets the same one-out-of-50
mass budget as the dense middle of the distribution, and a single location has to summarize a long,
heavy, sparsely-sampled tail. On a single 500k-step run, those tail locations are the noisiest things in
the head, and a noisy tail location feeds straight into the bootstrap mean and into the greedy argmax.
The seed-42-at-152 raggedness has the smell of exactly this: the tail of the return distribution is
under-resolved, so the mean estimate wobbles seed-to-seed. The fix I floated was: the classic-control
return range is *known* and bounded, so instead of learning *where* the mass sits, fix the locations and
learn the *mass* on them. That is the categorical representation, and it is worth deriving from the same
three requirements rather than just asserting it, because the loss and the projection are subtle and I
want them exactly right.

Same starting point, and I carry over the distributional Bellman recursion
`Z(s,a) =D R(s,a) + γ Z(s',a')` and its contraction story unchanged from QR-DQN: the operator
`γ`-scales the next-state distribution horizontally, shifts by the reward, and contracts in
**Wasserstein**, not in KL, total variation, or Kolmogorov distance, which are blind to the shrink.
QR-DQN's whole trick was to be Wasserstein-aware by learning locations. Now I am going to deliberately
give that up and pay for it, so I need to know exactly what I am buying.

What distribution class should the network output? I want multimodality — the win/lose bimodality of
LunarLander is the entire reason I left scalars — and I want it cheap. The natural move, borrowing from
how discrete models handle continuous values, is a **categorical** distribution on a *fixed* grid: pick
`N` canonical returns evenly spaced, `z_i = v_min + i·Δz`, `Δz = (v_max − v_min)/(N−1)`, and let the
network output a probability for each via a softmax, `Z_θ(s,a) = Σ_i p_i(s,a) δ_{z_i}`. This is
expressive — any shape on the grid, multimodal included — and trivial to compute (a softmax classifier
per action). And bounding the support to `[v_min, v_max]` is not just a concession, it is a *feature*
here: it bakes in the prior that returns beyond the range are all "equally extreme," which is a cleaner
inductive bias than nothing, and — the point that matters for my doubt — it lets me *spend* the `N` atoms
across the *value* axis at uniform spacing, so the crash tail gets honest grid resolution at its actual
return magnitude rather than one slippery quantile location. The cost is that I must supply `[v_min,
v_max]`, which QR-DQN didn't need; on classic control that is an acceptable price because the ranges are
known.

Two cheaper-looking alternatives deserve a few steps before I commit to fixing the grid, because
switching representations again is not free. The first is to just keep the quantile head and *raise* `N`
— push from 50 to 200 locations to resolve the tail. But think about where the extra locations go: the
quantile parametrization spends *equal mass* per location by construction, so quadrupling `N` quadruples
resolution *uniformly across the probability axis*, and the dense middle of the distribution — which was
never the problem — soaks up three-quarters of the new atoms. The crash tail, at a probability of maybe
a few percent, still gets only a couple of locations out of 200, and each is still a single sliding
number summarizing a long heavy tail. Worse, the all-pairs quantile loss is `O(N²)`, so `N = 200` is a
16× compute hit for a resolution gain that mostly lands where I do not need it. Raising `N` treats a
misallocation of resolution as if it were a shortage of resolution; it is the wrong axis. The second
alternative is a *parametric* head — output a mean and variance and model the return as Gaussian, or a
small mixture. That is cheap and tail-aware in principle, but a single Gaussian cannot represent the
win/lose bimodality that is the entire reason I left scalars, and a mixture needs me to fix the number of
modes and gives no closed-form distributional Bellman target — the shifted-and-scaled mixture is not a
mixture of the same family, so I would be back to a biased fit. Neither alternative addresses the actual
complaint. The complaint is that resolution should be allocated across the *value* axis, at the returns'
true magnitudes, not across the probability axis — and that is precisely what a fixed grid does.

Now the loss, and this is where the categorical representation forces a different path than QR-DQN.
Minimizing Wasserstein directly is out for the same reason as last time — the sampled `W_p` gradient is
biased. QR-DQN dodged that by switching the *loss* to quantile regression, whose minimizers are the
`W_1`-optimal locations; with fixed locations I cannot, the locations are not free. What I *can* minimize
unbiasedly from samples is **cross-entropy**, the softmax's native loss. The only obstruction is
geometric: the Bellman update scales each atom `z_j` by `γ` and shifts by `r`, so `r + γz_j` almost never
lands on the grid `{z_i}`. The target `TZ_θ` and my parametrization `Z_θ` then live on disjoint supports,
and a KL between disjoint supports is exactly the useless vertical quantity I have been avoiding. So I
must put the shifted target *back onto my grid* before taking cross-entropy.

The projection that respects the geometry: distribute each shifted atom's mass onto its two nearest grid
neighbors by linear interpolation, clamping anything outside `[v_min, v_max]` to the endpoints. The
shifted atom `T̂z_j = clamp(r + γz_j, v_min, v_max)` falls at fractional grid position
`b_j = (T̂z_j − v_min)/Δz ∈ [0, N−1]`, between `l = ⌊b_j⌋` and `u = ⌈b_j⌉`; the lower atom gets weight
`(u − b_j)` and the upper gets `(b_j − l)` (summing to 1, so mass is preserved), each times the source
probability `p_j`. Accumulating over all source atoms gives the projected target `m`. One subtlety I must
handle in code: when `b_j` is exactly an integer, `l = u` and the lower weight `(u − b_j) = 0` would drop
the mass, so I send the full `p_j` to that single atom — the `(l == u)` correction.

The projection has two edge behaviors I need to get right, so fix the grid — `v_min = −500`,
`v_max = 500`, `N = 51`, `Δz = 20`, `z_i = −500 + 20i` — and push atoms through. A generic source atom
`z_j = 0` with `r = 1`, `γ = 0.99` shifts to `T̂z = 1`, `b = (1 + 500)/20 = 25.05`, splitting mass
`0.95/0.05` onto `z_25, z_26`; that preserves both mass (`0.95 + 0.05 = 1`) and the first moment
(`0.95·0 + 0.05·20 = 1 = T̂z`), which is all "distribute by linear interpolation" should do. The integer
case is the one the `(l == u)` correction exists for: with `r = 0`, `T̂z = 0` gives `b = 25.0` exactly,
so `l = u = 25` and both naive weights `(u − b)` and `(b − l)` are zero — the entire `p_j` would silently
vanish. The correction rewrites the lower weight as `(u + [l==u] − b) = 1`, sending the full mass to atom
25. This is not a rare edge case: any `r + γz_j` landing on a multiple of `Δz` offset from `v_min` hits
it, and with `Δz = 20` and integer-ish rewards that happens constantly, so without the correction I would
bleed probability mass out of the target every batch. And the clamp: a bottom atom `z_j = −500` with
`r = −10` gives raw `−505`, below the floor, so `clamp` pins it to `−500` and the full mass piles onto the
floor atom — the bounded-support prior doing its job ("everything worse than −500 is equally
catastrophic"), not a bug. Terminal transitions
are handled by zeroing `γ` (the `(1 − dones)` factor), which collapses every shifted atom to `r`. The
target distribution is formed from the **target network** (frozen bootstrap, as DQN), and the greedy
action is chosen on the **mean** of the next-state distribution, `a* = argmax_a Σ_i z_i p_i(s',a)` —
keeping action selection a drop-in for epsilon-greedy DQN. The sample loss is the cross-entropy between
the projected target `m` and the prediction, `−Σ_i m_i log p_i(s,a)` — the cross-entropy term of
`KL(Φ T̂ Z_{θ⁻}(s,a) ‖ Z_θ(s,a))`. The distributional Bellman update has become **multiclass
classification** over the `N` atoms, which is exactly the unbiased-gradient regime I retreated to.

The unbiasedness is the whole justification for accepting a value-blind loss, so pin it down. The
cross-entropy against target `m` has gradient `p_i − m_i` in the logits. The subtlety in the bootstrap
setting: `m` is built from a *single sampled transition* `(s, a, r, s')`, so it is a random target
`m(r, s')`, and the gradient I descend is `p_i − m_i(r,s')` with expectation `p_i − E_{r,s'}[m_i]`.
Because both the projection `Φ` and the expectation are linear in the distribution, `E[m]` is `ΦT̂`
applied to the true next-state distribution, so the expected gradient drives `p` toward the *true*
projected Bellman target — no reshuffling of sample-to-atom matching, unlike the transport in
Wasserstein. That is how fixing the locations buys back an unbiased sampled gradient at the cost of
Wasserstein-awareness: a horizontal-metric loss I cannot descend traded for a vertical-metric one I can,
with the projection the bridge that makes the vertical loss meaningful despite the `γ`-shift.

A word on what I am trading versus QR-DQN, because it is the crux of whether this beats the last step.
KL is insensitive to the atom *values* — it only matches mass — so this loss is *not* Wasserstein-aware
the way the quantile loss was; if `v_min, v_max` are badly chosen the fixed grid can be a real handicap,
and the projection introduces a small bias the quantile approach avoided. What I get back is that the
mass on the crash tail is now a softmax probability on a grid point sitting at the tail's actual return
magnitude, learned by stable cross-entropy, rather than a single sliding location that has to *be* the
tail quantile. For a heavy-tailed bimodal return on a bounded-range task, that is the right trade — and
it is exactly the under-resolved tail I diagnosed in QR-DQN's ragged LunarLander seeds.

Now land it in this task's edit surface, with the scaffold-specific choices made explicit. The torso is
the **fixed MLP encoder** (`obs_dim → 120 → 84`), so `QNetwork` only changes the head: a linear map
`84 → |A|·N` reshaped to `(batch, |A|, N)` and softmaxed over the atom axis; `forward` returns the
per-action mean `Σ_i z_i p_i` so the harness argmaxes a clean `(batch, |A|)`. I set **`N = 51`** atoms.
The support is the load-bearing scaffold choice: I set **`v_min = −500`, `v_max = +500`** — *not* the
narrow `[−10, 10]` the generic recipe uses, because that range was tuned for clipped-reward Atari, and
here the returns are unclipped classic-control returns that genuinely span this range (CartPole reaches
+500, LunarLander runs roughly −400 to +300, Acrobot roughly −500 to −60). A grid that did not cover
−500 to +500 would clamp the crash tail and the CartPole cap to the endpoints and defeat the entire
point of switching to a fixed grid. `Δz = 1000/50 = 20`, so each atom is 20 return-units wide — coarse
in absolute terms, but the bimodal structure I care about (a few-hundred-point landing peak vs a
few-hundred-negative crash peak) is comfortably resolved at that spacing, and the returns that carry the
signal fall neatly on the grid: CartPole's cap of exactly `+500` is the top atom `z_50` and needs no
interpolation, so the coarse `Δz` cannot blur the solved task; a `+200` landing and a `−200` crash land
on atoms 35 and 15, distinct and 20 atoms apart, so the two LunarLander modes get a clean probability
lump each rather than smearing together. The grid is coarse but aligned. The projection is the
linear-interpolation operator with the `(l == u)` integer-`b` correction, terminals via `(1 − dones)`,
target built from the target net, greedy on the mean. I drop both the dueling head and the quantile head:
this step isolates the categorical-distribution effect, and the loss is cross-entropy, not quantile
Huber. Adam at the scaffold `lr`. (The full scaffold module is in the answer.)

So, against QR-DQN's {500, 197.15, −80.07}: CartPole should stay near 500, the only risk being the
coarse `Δz = 20` grid blurring the cap slightly if the head is badly resolved. Acrobot I expect on par,
low −80s, perhaps a touch noisier since its return sits in the dense lower-middle of a wide grid where
20-unit atoms are relatively coarse — any small drift below QR-DQN's level would be the grid-coarseness
cost showing up where I would predict it. LunarLander is the test: if the under-resolved tail was really
what kept QR-DQN's mean at 197 with ragged seeds, the fixed-grid mass model should lift the **mean above
197** and *tighten* the worst seed, because the crash tail now has honest grid resolution at its true
magnitude. If instead LunarLander comes back *lower* than QR-DQN, the diagnosis is unambiguous and points
past this whole family: the `[−500, 500]` grid with 20-unit atoms is too coarse for LunarLander's
bimodality, the KL loss's value-blindness has cost more than the quantile loss's tail-noise did, and the
next move would be to combine the best of both. The measured LunarLander mean against 197.15 is what
decides it.
