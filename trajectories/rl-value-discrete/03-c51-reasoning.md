The quantile head did the thing I bet on, and the LunarLander column proves it. The worst seed climbed
clean out of the crash basin: where the dueling head had {127.35, 229.07, **−89.25**} for a mean of 89,
QR-DQN posted {152.64, 243.81, **194.99**} for a mean of 197.15 — the negative seed is gone, the spread
collapsed from a 318-point range to a 91-point range, and every seed is now solidly positive. That is
exactly the prediction: modeling the return distribution let the greedy policy stop being fooled by a
high mean that hid crash mass. CartPole pinned at a clean 500 on all three seeds — the 50-way head did
not destabilize the already-solved task, which was the risk I flagged. Acrobot landed at −80.07, a hair
better than dueling's −82.6 and right in the low −80s as expected, since its return is nearly unimodal
and there is little distributional structure for quantiles to exploit. So the rung worked. But I closed
the last step with a specific doubt, and the LunarLander mean of 197 is exactly where that doubt lives:
197 is good, not great, and the per-seed numbers are still ragged — seed 42 at 152 is well below seed 123
at 244. A consistently-landing LunarLander agent should be scoring in the 200s on every seed. The
quantile picture of the return is still coarse where it matters most.

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

Same starting point: I refuse to keep collapsing the return to its mean, and I keep the distributional
Bellman recursion `Z(s,a) =D R(s,a) + γ Z(s',a')` that QR-DQN already relied on. The recursion exists.
The contraction story is the same and it is still the whole game: the operator `γ`-scales the next-state
distribution (a *horizontal* shrink toward 0), shifts by the reward, and mixes over transitions, so it
contracts in **Wasserstein** — `d_p` sees the horizontal slide and scales with `γ` — and emphatically
*not* in KL, total variation, or Kolmogorov distance, which compare mass at matched locations and are
blind to the shrink. QR-DQN's whole trick was to be Wasserstein-aware by learning locations. Now I am
going to deliberately give that up and pay for it, so I need to know exactly what I am buying.

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

Now the loss, and this is where the categorical representation forces a different path than QR-DQN. From
the theory, Wasserstein is *the* metric, so the first instinct is to minimize `d_p(TZ_θ, Z_θ)` directly.
But I already know — it is what pushed me to quantiles last rung — that the sampled Wasserstein gradient
is **biased**: with a mixture target seen one sample at a time, the partition inequality
`d_p^p(P, Q) ≤ E_I d_p^p(P_i, Q)` is strict, so SGD on the sampled `W_p` descends an upper bound whose
minimizer is the wrong point. (The clean counterexample: target `½δ_0 + ½δ_1`, prediction
`pδ_0 + (1−p)δ_1`; the true `d_1 = |p − ½|` is minimized at `p = ½`, but the expected *sampled* distance
is `½(1−p) + ½p = ½`, constant in `p`, gradient zero everywhere.) QR-DQN dodged this by switching the
*loss* to quantile regression, whose minimizers happen to be the `W_1`-optimal locations and whose
gradient is sign-only and unbiased. With fixed locations I cannot do that — the locations are not free.
What I *can* minimize unbiasedly from samples is **cross-entropy**, the softmax's native loss. The only
obstruction is geometric: the Bellman update scales each atom `z_j` by `γ` and shifts by `r`, so the
updated atom `r + γz_j` almost never lands on the grid `{z_i}`. The target `TZ_θ` and my parametrization
`Z_θ` live on disjoint supports, and a KL between disjoint supports is exactly the useless vertical
quantity I have been avoiding. So I must put the shifted target *back onto my grid* before taking
cross-entropy.

The projection that respects the geometry: distribute each shifted atom's mass onto its two nearest grid
neighbors by linear interpolation, clamping anything outside `[v_min, v_max]` to the endpoints. The
shifted atom `T̂z_j = clamp(r + γz_j, v_min, v_max)` falls at fractional grid position
`b_j = (T̂z_j − v_min)/Δz ∈ [0, N−1]`, between `l = ⌊b_j⌋` and `u = ⌈b_j⌉`; the lower atom gets weight
`(u − b_j)` and the upper gets `(b_j − l)` (summing to 1, so mass is preserved), each times the source
probability `p_j`. Accumulating over all source atoms gives the projected target `m`. One subtlety I must
handle in code: when `b_j` is exactly an integer, `l = u` and the lower weight `(u − b_j) = 0` would drop
the mass, so I send the full `p_j` to that single atom — the `(l == u)` correction. Terminal transitions
are handled by zeroing `γ` (the `(1 − dones)` factor), which collapses every shifted atom to `r`. The
target distribution is formed from the **target network** (frozen bootstrap, as DQN), and the greedy
action is chosen on the **mean** of the next-state distribution, `a* = argmax_a Σ_i z_i p_i(s',a)` —
keeping action selection a drop-in for epsilon-greedy DQN. The sample loss is the cross-entropy between
the projected target `m` and the prediction, `−Σ_i m_i log p_i(s,a)` — the cross-entropy term of
`KL(Φ T̂ Z_{θ⁻}(s,a) ‖ Z_θ(s,a))`. The distributional Bellman update has become **multiclass
classification** over the `N` atoms, which is exactly the unbiased-gradient regime I retreated to.

A word on what I am trading versus QR-DQN, because it is the crux of whether this rung beats the last.
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
few-hundred-negative crash peak) is comfortably resolved at that spacing. The projection is the
linear-interpolation operator with the `(l == u)` integer-`b` correction, terminals via `(1 − dones)`,
target built from the target net, greedy on the mean. I drop both the dueling head and the quantile head:
this rung isolates the categorical-distribution effect, and the loss is cross-entropy, not quantile
Huber. Adam at the scaffold `lr`. (The full scaffold module is in the answer.)

So, falsifiably, against QR-DQN's {500, 197.15, −80.07}: CartPole should stay pinned near 500 — the only
risk is the coarse `Δz = 20` grid blurring the cap, so I am watching that it does not slip below ~485 the
way a badly-resolved head might. Acrobot I expect on par again, low −80s, perhaps a touch noisier because
its return sits in the dense lower-middle of a wide grid where 20-unit atoms are relatively coarse — if
it drifts a couple of points below QR-DQN's −80, that is the grid-coarseness cost showing up exactly
where I would predict it. LunarLander is the test: if the under-resolved tail was really what kept
QR-DQN's mean at 197 with ragged seeds, the fixed-grid mass model should lift the **mean above 197** and
*tighten* the worst seed, because the crash tail now has honest grid resolution at its true magnitude.
That is the bar this rung must clear to earn the top of the ladder. If instead LunarLander comes back
*lower* than QR-DQN, the diagnosis is unambiguous and points past this whole family: the `[−500, 500]`
grid with 20-unit atoms is too coarse for LunarLander's bimodality, the KL loss's value-blindness has
cost more than the quantile loss's tail-noise did, and the right next move would be to combine the
best of both — but that is a different rung. For now this is the strongest fill the scaffold allows me to
isolate cleanly, and the measured LunarLander mean against 197.15 is exactly what decides it.
