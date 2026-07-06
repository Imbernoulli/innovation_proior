TAID confirmed the split I was watching for, and it points straight at the next move. `baseline:taid`
reads GSM8K 0.4685, MATH-500 0.280, AMC 0.0719. Difference it in problems. GSM8K went 599 → 618, a
gain of *nineteen* problems over rs_kd against a `±18`-problem standard error — the first move on the
whole ladder that clears the noise band, and the largest single-rung gain so far (rs_kd bought four,
dagger was the floor). That is exactly what the moving target was supposed to buy: the student walked
up from a reachable near-self target to the teacher instead of being asked to match it cold, and the
reachability fix paid off where it could. But MATH-500 went the *wrong* way: 0.280 is 140 of 500,
down six problems from rs_kd's 146 and five from dagger's 145 — below *both* fixed-target losses. Six
problems is inside the `±10`-problem MATH standard error, so I will not call it a large regression,
but the *direction* is unambiguous and it is precisely the falsifiable prediction I made coming true:
TAID fixed *reachability* but the loss it reaches the teacher *through* is still forward KL,
mass-covering, and on the hardest set — where the student most needs to commit to a coherent reasoning
mode rather than spread thin — mass-covering hurts, and the better it reaches an unreachable-in-detail
teacher, the more thinly it spreads. So the diagnosis is now sharp and it is no longer about the
target or the data: it is about the *divergence direction*. Three rungs have all used forward KL in
some guise (dagger's hard target is its argmax limit, RS-KD's top-K KL, TAID's KL to the intermediate
teacher), and the one place none of them touched is which way the KL points. That is what I attack now,
and the clean experimental design from last rung pays off: because TAID held the direction fixed and
changed only the target, the GSM8K-up / MATH-down split reads directly as "reachability is fixed,
direction is not." I note in passing that TAID's train loss was 0.363, between rs_kd's 0.342 and
dagger's 0.507, but I do not read anything into that ordering: TAID's objective is a KL against a
*moving* intermediate target, a third distinct thing to measure, so its train-loss number is no more
comparable to the others' than theirs were to each other. The accuracies are what carry across losses,
and the accuracies say direction.

There is a genuine puzzle buried in the taid number that I want to resolve before I move, because
resolving it is what turns "direction" from a guess into the diagnosis. Taid's schedule ends at
`t = 1`, where the intermediate teacher *is* the genuine teacher, reached through forward KL — which
is the same final target and the same divergence direction that rs_kd and the scaffold default used.
So at convergence taid is optimizing an objective almost indistinguishable from rs_kd's: forward KL
against the true teacher. Why, then, did taid's MATH-500 land at 0.280, *below* rs_kd's 0.292 and
dagger's 0.290, when all three end at forward KL to the same teacher? If the target and the direction
are identical at the end, then something about *how well the student actually reached that target*
must be doing the damage. And that is exactly what separates them. Rs_kd's GSM8K barely cleared dagger
(599 vs 595), so its student never got close to representing the teacher — it stayed a weakly-fit
model far from the teacher's distribution, and a distribution that is far from the teacher is not yet
*covering* anything, it is just under-trained. Taid's walk-up bought `+19` GSM8K problems and put the
student in the teacher's neighborhood for the first time — and it is precisely *there*, once the
student is close enough to spread its mass across the teacher's modes and long tail, that the
mass-covering pull of forward KL does its damage. So the two failures interact multiplicatively:
mass-covering only smears a student that has gotten close enough to smear, and rs_kd's under-reaching
had been *hiding* the direction problem all along. Fixing reachability with the moving target is what
unmasked it. That is why the MATH-500 regression is not evidence against the curriculum — the
curriculum plainly worked, GSM8K proves it — but evidence that, with reachability solved, the last
thing wrong is the direction, and it is now the only thing standing between me and the hard set.

Let me state the asymmetry precisely, because it is the whole game. Forward KL `KL(p_T ‖ p_S) =
Σ_v p_T(v)·log(p_T(v)/p_S(v))` weights each vocabulary token by the *teacher's* probability, so as
`p_S(v) → 0` wherever `p_T(v) > 0` the log term `log(p_T/p_S) → +∞` and the penalty blows up — it
forces the student to put *some* mass everywhere the teacher does. Mass-covering. If the student had
the teacher's capacity that would be fine; it could match the whole distribution. It does not. A 0.5B
student cannot represent every mode of a 7.6B math model's next-token distribution, so "cover all the
teacher's modes" makes it spread its limited mass thin — over the teacher's long-tail, low-probability
tokens included — and the result at free-run generation is a smeared, hedge-everything distribution
that samples incoherent or hallucinated continuations. That is precisely the MATH-500 regression: TAID
got the student *to* the teacher, but the mass-covering match parks it in the valley between the
teacher's modes, which on a long competition chain is where reasoning goes to die.

The opposite is reverse KL `KL(p_S ‖ p_T) = Σ_v p_S(v)·log(p_S(v)/p_T(v))`, weighted by the
*student's* own probability, so the penalty is large exactly where the student puts mass the teacher
finds unlikely. To drive it down the student withdraws mass from anything the teacher dislikes —
zero-forcing, mode-seeking: it concentrates on the teacher's major modes and abandons the tail.
Picture fitting a single Gaussian to a two-bump mixture. Forward KL, weighted by the target, must
account for both bumps, so it parks the single mode in the valley between them to keep some mass under
each — the incoherent average I just diagnosed. Reverse KL, weighted by the model's own mass, pays
nothing for the region it vacates, so it snaps onto one bump and commits, ignoring the other. For
compression under capacity mismatch, committing to one coherent teacher behavior is exactly what the
MATH-500 failure says I want. So the direction I should move is toward reverse KL.

Two limits make the asymmetry exact and settle that this is a real mechanism, not a metaphor about
Gaussians. Ask what each direction charges the student for *abandoning* a teacher mode — driving
`p_S(v) → 0` on some token `v` where `p_T(v) > 0`. Under reverse KL the token's contribution is
`p_S·log(p_S/p_T)`, and `lim_{p_S→0} p_S·log(p_S/p_T) = 0` because `x log x → 0`; so vacating a mode is
*free* — the student pays nothing to zero out mass, which is why reverse KL will happily abandon the
teacher's minor modes and concentrate. Under forward KL the same token's contribution is
`p_T·log(p_T/p_S)`, and `lim_{p_S→0} p_T·log(p_T/p_S) = +∞`; so vacating a mode is *infinitely
costly* — the student is forbidden to zero out any token the teacher likes, which is why forward KL
forces the smear. Free-to-vacate versus infinite-to-vacate is the entire difference between
mode-seeking and mass-covering, and it is a two-line limit, not a picture. The MATH-500 regression is
"forced smear," and the cure is to move toward the direction where vacating is free.

Before I reach for a new divergence I should kill the cheap shortcut, because if it worked I would not
need to touch the loss family at all. The tempting one is temperature: sharpen the student — divide
its logits by a temperature below one so `p_S` becomes peakier — and call that "committing to a mode."
It does not buy mode-seeking, and the two limits I just wrote say why. Mode-seeking is a statement
about *which distribution weights the sum* — reverse KL weights by `p_S`, forward by `p_T` — not about
how sharp either distribution is. Sharpening `p_S` under *forward* KL changes the student's entropy but
leaves the objective weighted by `p_T`, so the infinite penalty for vacating a teacher mode,
`lim_{p_S(v)→0} p_T(v)·log(p_T(v)/p_S(v)) = +∞`, is untouched — a sharper student is still forbidden to
zero out any token the teacher likes, so it still smears to cover them, just with a peakier shape
around each mode it is forced to keep. And the scaffold applies temperature *symmetrically* to both
logit tensors anyway, so it cannot even tilt the weighting toward one side. Temperature moves sharpness,
not the mass-covering-versus-mode-seeking character; it is orthogonal to the direction axis. So the
shortcut is a mirage, and I do have to change the divergence itself.

But I do not want to slam all the way to pure reverse KL in one step, for two reasons I can make
precise. First, reverse KL overshoots into the opposite ditch for a low-capacity student — it can
collapse onto one or two dominant modes and drop the teacher's structure entirely, trading
mass-covering mush for mode-collapse brittleness. Second, raw KL in either direction is *unbounded*
when the two distributions have near-disjoint support, which happens constantly early in training when
the student is still far off; the `log(p/q)` with a vanishing denominator produces enormous,
destabilizing gradients. I want a one-parameter family that has forward KL at one end and reverse KL
at the other, behaves sensibly in between, and stays *bounded* even on disjoint supports — so I can
sit at a balanced interior point rather than commit to an endpoint.

There is more than one such family, and picking the right one is not automatic, so let me rule out the
obvious competitor before I reach for the mixture. The most naive interpolator is a convex combination
of the two raw KLs, `α·KL(p_T ‖ p_S) + (1−α)·KL(p_S ‖ p_T)`, which is forward at `α = 1`, reverse at
`α = 0`, and something in between otherwise. But it fails the second requirement outright: both legs
still have a *raw* distribution in the denominator, so on disjoint support both `KL(p_T ‖ p_S)` and
`KL(p_S ‖ p_T)` are individually infinite, and any positive-weight combination of two infinities is
infinite. The naive combo interpolates the direction but inherits the early-training blow-up from both
sides — it does not tame anything. What I need is a family whose denominators are *floored*, and the
way to floor them is to divide by a *mixture* that always contains some of both distributions rather
than by a raw one.

The Jensen-Shannon divergence is exactly that family, because it is built from a mixture. Introduce a
coefficient `β ∈ (0,1)`, form `M = β·p_T + (1−β)·p_S`, and define the generalized JSD as
`D_JSD^β(p_T ‖ p_S) = β·KL(p_T ‖ M) + (1−β)·KL(p_S ‖ M)`. Now the denominator in each KL is `M`, which
carries mass wherever *either* distribution does, so the logs cannot diverge — the floor I wanted is
structural. Let me verify it actually interpolates the two directions, because the boundary behavior
is subtle: the raw `D_JSD^β` itself goes to zero as `β → 0` or `β → 1` (both weights and the mixture
collapse), so the KLs appear as *scaled* limits, not literal endpoint values, and I want to see the
forward KL fall out. Take `β → 0`, so `M = p_S + β·(p_T − p_S)`. The second term first:
`KL(p_S ‖ M) = Σ p_S·log(p_S/M) = −Σ p_S·log(1 + β(p_T−p_S)/p_S)`, and expanding the log to first
order gives `−β·Σ(p_T − p_S) + O(β²) = −β·(1 − 1) + O(β²) = O(β²)`, so `(1−β)·KL(p_S ‖ M)` is
second-order in `β` and vanishes faster than the first term. The first term: as `β → 0`, `M → p_S`, so
`KL(p_T ‖ M) → KL(p_T ‖ p_S)`, and `β·KL(p_T ‖ M) = β·KL(p_T ‖ p_S) + O(β²)`. Divide through:
`lim_{β→0} D_JSD^β / β = KL(p_T ‖ p_S)` — the forward-KL direction, recovered exactly. By the symmetry
`D_JSD^β(p_T ‖ p_S) = D_JSD^{1−β}(p_S ‖ p_T)` (swap `β ↔ 1−β` and `p_T ↔ p_S` and `M` is unchanged),
the `β → 1` end gives reverse KL. So small `β` is mass-covering, large `β` is mode-seeking, and
`β = 0.5` is the symmetric Jensen-Shannon divergence sitting exactly between.

And now the boundedness, which I claimed and should actually check on the worst case. Suppose `p_T` and
`p_S` have *disjoint* support — the pathological early-training case where raw KL is infinite. On the
tokens where `p_T > 0`, the mixture is `M = β·p_T` (the `p_S` leg is zero there), so
`KL(p_T ‖ M) = Σ p_T·log(p_T/(β·p_T)) = Σ p_T·log(1/β) = −log β`. Symmetrically `KL(p_S ‖ M) = −log(1−β)`
on the student's support. So `D_JSD^β = β·(−log β) + (1−β)·(−log(1−β)) = H(β)`, the binary entropy of
`β`, which is finite for all `β ∈ (0,1)` and maxes at `log 2` when `β = 0.5`. So on the exact case
that sends forward and reverse KL to infinity, symmetric JSD returns `log 2 ≈ 0.693` — a bounded,
well-behaved number. That is the second gift I did not have to go looking for: the mixture denominator
that floors the logs also caps the whole divergence, so the early-training blow-up that pure reverse
KL (and the naive raw-KL combo) would risk is tamed automatically. This is exactly why JSD is the
*right* interpolator and not merely *an* interpolator — it is the one that satisfies both requirements
at once.

I claimed the *gradient* is bounded too, and I should actually see it, because a bounded loss value
and a bounded gradient are different claims and it is the gradient that trains. Differentiate
`D_JSD^{0.5}` in the student logits: only the pieces that depend on `p_S` survive — the `KL(p_S ‖ M)`
term and the student leg inside `M` in both KLs — and after the algebra the per-token coefficient that
multiplies the softmax Jacobian is a combination of `log(p_S/M)` and terms in the ratios `p_S/M` and
`p_T/M`. Every one of those ratios is floored by the mixture: `M = 0.5 p_T + 0.5 p_S ≥ 0.5 p_S` gives
`p_S/M ≤ 2`, and `M ≥ 0.5 p_T` gives `p_T/M ≤ 2`, so `log(p_S/M) ≤ log 2` and each ratio piece is at
most `2` — *everywhere*, including on the disjoint-support tokens where the raw reverse-KL coefficient
`log(p_S/p_T)` runs to `+∞`. The same mixture denominator that caps the loss value at `log 2` caps the
gradient coefficient at `log 2`, structurally, with nothing to tune. That is the property I actually
need: it is the gradient, not the loss value, that would otherwise emit the destabilizing
early-training steps, and it is bounded by construction.

The boundedness has a concrete training consequence I want to bank, because it decides what
scaffolding I need around the loss. Since `D_JSD^{0.5} ≤ log 2` and its gradient is correspondingly
bounded even when the student and teacher momentarily disagree completely, the loss cannot emit the
enormous early-training steps that a raw reverse KL — or the naive raw-KL combo — would produce on
near-disjoint support. So I can sit at the interior *without* bolting on gradient clipping, a loss
warmup, or the other stabilizers a raw mode-seeking objective would force me to add, none of which the
single-loss edit surface even lets me configure cleanly. The mixture buys stability structurally, in
the loss itself, rather than in trainer plumbing I do not own. Let me put one number on the interior
value so "half and half" is not just a slogan. Take a token where the teacher is confident on the
right continuation, `p_T = (0.8, 0.1, 0.1)`, and the student is still confident on a wrong one,
`p_S = (0.1, 0.1, 0.8)`. The mixture is `M = 0.5(p_T + p_S) = (0.45, 0.1, 0.45)`. Then
`KL(p_T ‖ M) = 0.8·log(0.8/0.45) + 0.1·log(0.1/0.1) + 0.1·log(0.1/0.45) = 0.8·0.575 + 0 + 0.1·(−1.504)
= 0.460 − 0.150 = 0.310`, and by symmetry `KL(p_S ‖ M) = 0.310`, so `D_JSD^{0.5} = 0.310` — a
moderate, finite penalty. The raw forward KL on this same pair is `0.8·log8 + 0 + 0.1·log(1/8) =
0.8·2.079 − 0.1·2.079 = 1.455`, and the raw reverse KL is the mirror `1.455`; both are almost five
times the JSD. So even where student and teacher badly disagree, the interior divergence returns a
controlled gradient signal instead of a spike — exactly the behavior I want on the ragged
early-training positions that the on-policy rollouts keep producing.

This is also the move that subsumes the whole ladder so far, and seeing that convinces me it is the
right generalization rather than a fourth ad-hoc option. The general object behind every rung is: pick
a divergence `D` on the forward↔reverse family, and pick how much data is the student's own (the
trainer's `lmbda`). Off-policy supervised KD is the forward-KL corner; the scaffold default is forward
KL; RS-KD is forward KL over a sparse support; even TAID's forward KL to a moving target is a
forward-KL-flavored point with a curriculum bolted on. None of them moved off the forward-KL face of
the family — they varied the *target* and the *support*, never the *direction*. Generalized JSD at
`β = 0.5` steps directly onto the interior of the divergence axis for the first time — half
mass-covering, half mode-seeking — which is exactly the medicine the MATH-500 regression prescribed:
pull the student off the pure mass-covering match without slamming it into mode-collapse.

Now the per-token loss in code, because the family has endpoints the interior formula cannot touch and
the KL direction is a silent trap. For the interior `β = 0.5` I need `log M` where `M = β·p_T +
(1−β)·p_S`. Computing it as `log((1−β)·exp(log p_S) + β·exp(log p_T))` would underflow when the
log-probs are very negative, so I form it as a log-sum-exp of the two shifted log-prob tensors:
`log M = logsumexp([log p_S + log(1−β), log p_T + log β])` stacked along a new axis. That gets `log M`
from the two log-prob tensors without ever leaving log space, and it is stable because logsumexp
subtracts the running max internally. Then the framework's `kl_div(input=log_q, target=log_p,
log_target=True)` computes `KL(p ‖ q)` — it treats the *input* as the log of the denominator and the
*target* as the distribution the KL is from. I want `KL(p_T ‖ M)`, so the mixture log-probs are the
input and the teacher log-probs are the target; likewise `KL(p_S ‖ M)` is mixture-input,
student-target. Get this backwards and I would minimize a quietly wrong objective with no error
raised. Combine `per_token = β·KL(p_T ‖ M) + (1−β)·KL(p_S ‖ M)`, mask to completion tokens, reduce per
token. The temperature divides both logit tensors before the softmax, consistent with the rest of the
ladder, and shared so it measures behavior rather than a sharpness mismatch.

One cost check, because I am now computing more per token than any rung below me. Symmetric JSD needs
two `log_softmax` passes, one `logsumexp` to form `log M`, and two `kl_div` reductions — roughly three
full-vocabulary passes over the `[B, T, V]` tensors against forward KL's one. But the memory footprint
is unchanged: `log M` is one more `[B, T, V]` tensor of exactly the width the scaffold already
materializes, no wider, and the extra elementwise vocab arithmetic is negligible against the two model
forwards that produced the logits in the first place. So the interior divergence costs a little more
arithmetic and no more memory than the forward-KL default — affordable, and cheaper than
three-KLs-per-token sounds.

I should be precise about what this task's loss does and does not implement, because the method's full
form exposes the divergence as a tunable knob and this rung pins it. The general GKD object is two
axes — the data fraction and any divergence on the family — and its loss is usually written with
explicit `β = 0` and `β = 1` endpoint branches (forward and reverse KL) plus the interior JSD branch,
so a trainer can sweep `β`. This task's baseline does *not* expose that sweep: it hard-pins
`beta_use = 0.5` and always computes the genuine interior generalized JSD via the mixture, ignoring the
`beta` argument the signature passes in. So what I am landing is specifically the *symmetric
Jensen-Shannon* point of GKD — not the configurable family, not an endpoint — the canonical
`β = 0.5` setting. The data axis stays the trainer's static `lmbda` mixing, untouched, exactly as on
every prior rung; the only thing this rung changes versus TAID is the divergence, from forward KL to
symmetric JSD. The full scaffold body is in the answer.

Falsifiable expectations against TAID and the cluster below it. Symmetric JSD is half mode-seeking, so
the prediction is asymmetric across the two reliable metrics, and I can be quantitative about it. On
MATH-500 I expect the *recovery* that TAID's forward KL gave up: pulling the student off pure
mass-covering should let it commit more to the teacher's reasoning modes, so MATH-500 should climb back
above the dagger/RS-KD level (0.290/0.292) and above TAID's 0.280 — into the low-0.31s, a recovery of
roughly ten to fifteen problems from TAID, which would clear the `±10`-problem noise band. On GSM8K,
where TAID's reachability fix already got to 0.4685 (618 problems) and short arithmetic chains are more
forgiving of a mass-covering match, I expect a smaller move — into the low-0.47s, a handful of
problems over TAID — because the divergence fix helps most exactly where mass-covering hurt most, and
that was MATH-500, not GSM8K. So the signature I am predicting is GSM8K a touch over TAID and MATH-500
a clear recovery, which would make this the strongest *interior-divergence* rung. If instead GSM8K
*jumped* and MATH-500 stayed flat, that would falsify the mass-covering diagnosis — it would say
mode-seeking helps the easy set, not the hard one, which is the opposite of the mechanism I derived.
The open question this rung leaves for the top of the ladder is whether `β = 0.5` is far enough toward
mode-seeking: if half-JSD recovers MATH-500 but GSM8K only inches up, the natural next move is to push
the divergence all the way to the reverse-KL endpoint and see whether full mode-seeking on the
student's own rollouts beats the balanced compromise. AMC stays noise; I will not read it.
