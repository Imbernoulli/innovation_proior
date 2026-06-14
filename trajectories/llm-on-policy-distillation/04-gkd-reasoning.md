TAID confirmed the split I was watching for, and it points straight at the next move. `baseline:taid`
reads GSM8K 0.4685, MATH-500 0.280, AMC 0.0719. On GSM8K that is a real jump — +0.014 over rs_kd's
0.4541, +0.017 over dagger — the largest single-rung gain on the ladder so far, and exactly what the
moving target was supposed to buy: the student walked up from a reachable near-self target to the
teacher instead of being asked to match it cold, and the reachability fix paid off where it could. But
MATH-500 went the *wrong* way: 0.280, *below* both rs_kd (0.292) and dagger (0.290). That is the
falsifiable prediction I made coming true — TAID fixed *reachability* but the loss it reaches the
teacher *through* is still forward KL, mass-covering, and on the hardest set, where the student most
needs to commit to a coherent reasoning mode rather than spread thin, mass-covering hurts. So the
diagnosis is now sharp and it is no longer about the target or the data: it is about the *divergence
direction*. Three rungs have all used forward KL in some guise (dagger's hard target is its argmax
limit, RS-KD's top-K KL, TAID's KL to the intermediate teacher), and the one place none of them
touched is which way the KL points. That is what I attack now.

Let me state the asymmetry precisely, because it is the whole game. Forward KL `KL(p_T ‖ p_S) =
Σ_v p_T(v)·log(p_T(v)/p_S(v))` weights each vocabulary token by the *teacher's* probability, so as
`p_S(v) → 0` wherever `p_T(v) > 0` the term blows up — it forces the student to put *some* mass
everywhere the teacher does. Mass-covering. If the student had the teacher's capacity that would be
fine; it could match the whole distribution. It does not. A 0.5B student cannot represent every mode
of a 7.6B math model's next-token distribution, so "cover all the teacher's modes" makes it spread its
limited mass thin — over the teacher's long-tail, low-probability tokens included — and the result at
free-run generation is a smeared, hedge-everything distribution that samples incoherent or
hallucinated continuations. That is precisely the MATH-500 regression: TAID got the student *to* the
teacher, but the mass-covering match parks it in the valley between the teacher's modes, which on a
long competition chain is where reasoning goes to die.

The opposite is reverse KL `KL(p_S ‖ p_T) = Σ_v p_S(v)·log(p_S(v)/p_T(v))`, weighted by the
*student's* own probability, so the penalty is large exactly where the student puts mass the teacher
finds unlikely. To drive it down the student withdraws mass from anything the teacher dislikes —
zero-forcing, mode-seeking: it concentrates on the teacher's major modes and abandons the tail.
Picture fitting a single Gaussian to a two-bump mixture: forward KL parks it in the valley to cover
both (the incoherent average I just diagnosed), reverse KL snaps onto one bump and commits. For
compression under capacity mismatch, committing to one coherent teacher behavior is exactly what the
MATH-500 failure says I want. So the direction I should move is toward reverse KL.

But I do not want to slam all the way to pure reverse KL in one step, for two reasons. First, reverse
KL overshoots into the opposite ditch for a low-capacity student — it can collapse onto one or two
dominant modes and drop the teacher's structure entirely, trading mass-covering mush for mode-collapse
brittleness. Second, raw KL in either direction is *unbounded* when the two distributions have
near-disjoint support, which happens constantly early in training when the student is still far off;
that produces enormous, destabilizing gradients. I want a one-parameter family that has forward KL at
one end and reverse KL at the other, behaves sensibly in between, and stays *bounded* even on disjoint
supports — so I can sit at a balanced interior point rather than commit to an endpoint.

The Jensen-Shannon divergence is the natural such family because it is built from a mixture.
Introduce a coefficient `β ∈ (0,1)`, form `M = β·p_T + (1−β)·p_S`, and define the generalized JSD as
`D_JSD^β(p_T ‖ p_S) = β·KL(p_T ‖ M) + (1−β)·KL(p_S ‖ M)`. Why does it interpolate? At the boundary I
have to be careful, because the raw `D_JSD^β` itself goes to zero as `β → 0` or `β → 1`; the KLs
appear as *scaled* limits, not the literal endpoint values. As `β → 0`, `M → p_S`, the second term is
second-order in `β`, and `lim_{β→0} D_JSD^β/β = KL(p_T ‖ p_S)` — the forward-KL direction. By the
symmetry `D_JSD^β(p_T ‖ p_S) = D_JSD^{1−β}(p_S ‖ p_T)`, the `β → 1` end gives reverse KL. So small `β`
is mass-covering, large `β` is mode-seeking, and `β = 0.5` is the symmetric Jensen-Shannon divergence
sitting exactly between. And the second gift, which I did not go looking for: JSD is *bounded* even
when `p_T` and `p_S` are disjoint, whereas plain KL is infinite there — so the early-training blow-up
that pure reverse KL would risk is tamed automatically.

This is the move that subsumes the whole ladder so far, and seeing that convinces me it is the right
generalization rather than a fourth ad-hoc option. The general object behind every rung is: pick a
divergence `D` on the forward↔reverse family, and pick how much data is the student's own (the
trainer's `lmbda`). Off-policy supervised KD is the forward-KL corner; the scaffold default is forward
KL; RS-KD is forward KL over a sparse support; even TAID's forward KL to a moving target is a forward-
KL-flavored point with a curriculum bolted on. None of them moved off the forward-KL face of the
family. Generalized JSD at `β = 0.5` steps directly onto the interior of the divergence axis for the
first time — half mass-covering, half mode-seeking — which is exactly the medicine the MATH-500
regression prescribed: pull the student off the pure mass-covering match without slamming it into
mode-collapse.

Now the per-token loss in code, because the family has endpoints the interior formula cannot touch and
the KL direction is a silent trap. For the interior `β = 0.5` I need `log M` where `M = β·p_T +
(1−β)·p_S`. Computing it as `log((1−β)·exp(log p_S) + β·exp(log p_T))` would underflow, so I form it
as a log-sum-exp of the two shifted log-prob tensors: `log M = logsumexp([log p_S + log(1−β),
log p_T + log β])` stacked along a new axis. That gets `log M` from the two log-prob tensors without
ever leaving log space. Then the framework's `kl_div(input=log_q, target=log_p, log_target=True)`
computes `KL(p ‖ q)` — it treats the *input* as the log of the denominator and the *target* as the
distribution the KL is from. I want `KL(p_T ‖ M)`, so the mixture log-probs are the input and the
teacher log-probs are the target; likewise `KL(p_S ‖ M)` is mixture-input, student-target. Get this
backwards and I would minimize a quietly wrong objective with no error raised. Combine
`per_token = β·KL(p_T ‖ M) + (1−β)·KL(p_S ‖ M)`, mask to completion tokens, reduce per token. The
temperature divides both logit tensors before the softmax, consistent with the rest of the ladder.

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
the prediction is asymmetric across the two reliable metrics. On MATH-500 I expect the *recovery* that
TAID's forward KL gave up: pulling the student off pure mass-covering should let it commit more to the
teacher's reasoning modes, so MATH-500 should climb back above the dagger/RS-KD level — into the low-
0.31s — reversing TAID's 0.280 regression. On GSM8K, where TAID's reachability fix already got to
0.4685 and short arithmetic chains are more forgiving of a mass-covering match, I expect a smaller
move — into the low-0.47s — edging past TAID but not by the margin TAID gained over RS-KD, because the
divergence fix helps most exactly where mass-covering hurt most, and that was MATH-500, not GSM8K. So
the signature I am predicting is GSM8K ≈ 0.47 (a touch over TAID) and MATH-500 ≈ 0.31 (a clear
recovery), which would make this the strongest *interior-divergence* rung. The open question it leaves
for the top of the ladder is whether `β = 0.5` is far enough toward mode-seeking: if half-JSD recovers
MATH-500 but GSM8K only inches up, the natural next move is to push the divergence all the way to the
reverse-KL endpoint and see whether full mode-seeking on the student's own rollouts beats the
balanced compromise. AMC stays noise; I will not read it.
