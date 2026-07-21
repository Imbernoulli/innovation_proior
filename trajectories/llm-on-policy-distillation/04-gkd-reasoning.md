TAID confirmed the split I was watching for, and it points straight at the next move.
`baseline:taid` reads GSM8K 0.4685, MATH-500 0.280, AMC 0.0719. In problems: GSM8K went 599 → 618, a
gain of *nineteen* over rs_kd against a `±18`-problem standard error — the first move that clears the
noise band and the largest single-step gain so far. That is what the moving target was supposed to
buy: the student walked up from a reachable near-self target to the teacher instead of matching it
cold. But MATH-500 went the *wrong* way, 0.280 is 140 of 500, down six from rs_kd and five from
dagger — below *both* fixed-target losses. Six problems is inside the `±10`-problem MATH standard
error, so not a large regression, but the *direction* is unambiguous and it is exactly the prediction
I made: TAID fixed reachability but reaches the teacher *through* forward KL, mass-covering, and on
the hardest set — where the student most needs to commit to a coherent reasoning mode rather than
spread thin — mass-covering hurts, and the better it reaches an unreachable-in-detail teacher, the
more thinly it spreads. So the diagnosis is now about the *divergence direction*. Three losses have
all used forward KL in some guise (dagger's argmax limit, RS-KD's top-K KL, TAID's KL to the
intermediate teacher); the one place none touched is which way the KL points. TAID's train loss was
0.363, between rs_kd's and dagger's, but it is a KL against a *moving* target, a third incomparable
objective, so I read nothing into the ordering. The accuracies carry across losses, and they say
direction.

There is a puzzle buried in the taid number worth resolving, because it turns "direction" from a
guess into the diagnosis. TAID's schedule ends at `t = 1`, where the intermediate teacher *is* the
genuine teacher, reached through forward KL — the same final target and direction rs_kd and the
scaffold default used. So at convergence taid optimizes an objective almost indistinguishable from
rs_kd's, yet its MATH-500 landed at 0.280, below both rs_kd's 0.292 and dagger's 0.290. If the target
and direction are identical at the end, then how well the student actually *reached* that target must
be doing the damage. Rs_kd's GSM8K barely cleared dagger, so its student stayed far from the teacher's
distribution — under-trained, not yet *covering* anything. Taid's walk-up put the student in the
teacher's neighborhood for the first time, and it is precisely *there*, once the student is close
enough to spread its mass across the teacher's modes and long tail, that the mass-covering pull of
forward KL does its damage. The two failures interact multiplicatively: mass-covering only smears a
student that has gotten close enough to smear, and rs_kd's under-reaching had been *hiding* the
direction problem. Fixing reachability unmasked it — which is why the MATH-500 regression is evidence
not against the curriculum (GSM8K proves it worked) but that, with reachability solved, direction is
the last thing wrong.

State the asymmetry precisely, because it is the whole game. Forward KL `KL(p_T ‖ p_S) = Σ_v
p_T(v)·log(p_T/p_S)` weights each token by the *teacher's* probability, so as `p_S(v) → 0` where
`p_T(v) > 0` the penalty blows up — it forces the student to put *some* mass everywhere the teacher
does. Mass-covering. A 0.5B student cannot represent every mode of a 7.6B math model, so "cover all
the teacher's modes" spreads its limited mass thin, over the long-tail low-probability tokens
included, and free-run generation samples smeared, incoherent continuations. That is the MATH-500
regression: TAID got the student *to* the teacher, but the mass-covering match parks it in the valley
between the teacher's modes, where on a long chain reasoning goes to die. The opposite is reverse KL
`KL(p_S ‖ p_T)`, weighted by the *student's* own probability, so the penalty is large exactly where
the student puts mass the teacher finds unlikely; to drive it down the student withdraws mass from
anything the teacher dislikes — zero-forcing, mode-seeking, concentrating on the major modes. Fitting
a single Gaussian to a two-bump mixture: forward KL parks the mode in the valley to keep some mass
under each; reverse KL snaps onto one bump and commits.

Two limits make the asymmetry exact, not a metaphor. Ask what each direction charges for *abandoning*
a teacher mode, `p_S(v) → 0` where `p_T(v) > 0`. Under reverse KL the token's contribution is
`p_S·log(p_S/p_T) → 0` (since `x log x → 0`), so vacating a mode is *free*. Under forward KL it is
`p_T·log(p_T/p_S) → +∞`, so vacating is *infinitely costly*. Free-to-vacate versus infinite-to-vacate
is the entire difference between mode-seeking and mass-covering. The MATH-500 regression is forced
smear, and the cure is to move toward the direction where vacating is free.

Before reaching for a new divergence I kill the cheap shortcut: temperature. Sharpening the student —
dividing its logits by a temperature below one so `p_S` becomes peakier — does not buy mode-seeking,
because mode-seeking is a statement about *which distribution weights the sum* (reverse weights by
`p_S`, forward by `p_T`), not about how sharp either distribution is. Sharpening `p_S` under forward
KL leaves the objective weighted by `p_T`, so the infinite penalty for vacating a teacher mode is
untouched — a sharper student still smears to cover them, just peakier around each. And the scaffold
applies temperature *symmetrically* to both tensors anyway, so it cannot tilt the weighting.
Temperature moves sharpness, orthogonal to the direction axis; I have to change the divergence itself.

But not all the way to pure reverse KL in one step, for two reasons I can make precise. Reverse KL can
overshoot into the opposite ditch for a low-capacity student — collapse onto one or two dominant modes
and drop the teacher's structure entirely. And raw KL in either direction is *unbounded* when the
supports are near-disjoint, which happens constantly early in training, producing enormous
destabilizing gradients. I want a one-parameter family with forward KL at one end, reverse at the
other, sensible in between, and *bounded* even on disjoint supports, so I can sit at a balanced
interior point rather than commit to an endpoint.

The obvious competitor fails, so rule it out first. A convex combination of the two raw KLs
`α·KL(p_T ‖ p_S) + (1−α)·KL(p_S ‖ p_T)` is forward at `α = 1`, reverse at `α = 0` — but on disjoint
support both legs are individually infinite, and any positive-weight combination of two infinities is
infinite. It interpolates the direction but inherits the early-training blow-up from both sides. What
I need is *floored* denominators, and the way to floor them is to divide by a *mixture* that always
contains some of both distributions rather than a raw one.

The Jensen-Shannon divergence is exactly that, because it is built from a mixture. With `M = β·p_T +
(1−β)·p_S`, define `D_JSD^β = β·KL(p_T ‖ M) + (1−β)·KL(p_S ‖ M)`. The denominator `M` carries mass
wherever *either* distribution does, so the logs cannot diverge — the floor is structural. It
interpolates the directions, though the boundary is subtle: the raw `D_JSD^β` itself goes to zero as
`β → 0`, so the KLs appear as *scaled* limits. Take `β → 0`, `M = p_S + β(p_T − p_S)`: the
`KL(p_S ‖ M)` term expands to `−β·Σ(p_T − p_S) + O(β²) = O(β²)` and vanishes faster, while
`KL(p_T ‖ M) → KL(p_T ‖ p_S)`, so `lim_{β→0} D_JSD^β/β = KL(p_T ‖ p_S)`, forward KL recovered exactly.
By the symmetry `D_JSD^β(p_T ‖ p_S) = D_JSD^{1−β}(p_S ‖ p_T)`, the `β → 1` end gives reverse KL. So
small `β` is mass-covering, large `β` is mode-seeking, and `β = 0.5` is the symmetric Jensen-Shannon
point sitting exactly between.

Now boundedness on the worst case. If `p_T` and `p_S` have *disjoint* support, on the tokens where
`p_T > 0` the mixture is `M = β·p_T`, so `KL(p_T ‖ M) = Σ p_T·log(1/β) = −log β`, and symmetrically
`KL(p_S ‖ M) = −log(1−β)`, giving `D_JSD^β = β·(−log β) + (1−β)·(−log(1−β)) = H(β)`, the binary
entropy, finite for all `β ∈ (0,1)` and maxing at `log 2` when `β = 0.5`. So on the exact case that
sends forward and reverse KL to infinity, symmetric JSD returns `log 2 ≈ 0.693` — which is why JSD is
the *right* interpolator and not merely *an* interpolator. The *gradient* is capped too, and that is
the claim that actually matters: differentiating `D_JSD^{0.5}` in the student logits, every ratio is
floored by the mixture — `M ≥ 0.5 p_S` gives `p_S/M ≤ 2`, `M ≥ 0.5 p_T` gives `p_T/M ≤ 2` —
everywhere, including on the disjoint-support tokens where the raw reverse-KL coefficient
`log(p_S/p_T)` runs to `+∞`. So I can sit at the interior *without* bolting on gradient clipping or a
loss warmup the single-loss edit surface cannot configure cleanly; the mixture buys stability
structurally, in the loss itself.

This move also subsumes the progression so far, which convinces me it is the right generalization
rather than a fourth ad-hoc option. The general object is: pick a divergence on the forward↔reverse
family, and pick how much data is the student's own (the trainer's `lmbda`). The scaffold default is
forward KL, RS-KD is forward KL over a sparse support, TAID's forward KL to a moving target is a
forward-flavored point with a curriculum bolted on — none moved off the forward-KL face of the
family; they varied the *target* and the *support*, never the *direction*. Generalized JSD at `β =
0.5` steps onto the interior of the divergence axis for the first time — half mass-covering, half
mode-seeking — exactly the medicine the MATH-500 regression prescribed: pull the student off pure
mass-covering without slamming it into mode-collapse.

The per-token loss in code, where the KL direction is a silent trap. For `β = 0.5` I need `log M` with
`M = β·p_T + (1−β)·p_S`; computing it as `log((1−β)·exp(log p_S) + β·exp(log p_T))` underflows on very
negative log-probs, so I form `log M = logsumexp([log p_S + log(1−β), log p_T + log β])` stacked along
a new axis — stable, never leaving log space. Then the framework's `kl_div(input=log_q, target=log_p,
log_target=True)` computes `KL(p ‖ q)`, treating the *input* as the log-denominator and the *target*
as the distribution the KL is from. For `KL(p_T ‖ M)` the mixture log-probs are the input and the
teacher log-probs the target; likewise `KL(p_S ‖ M)` is mixture-input, student-target. Backwards, I
would minimize a quietly wrong objective with no error raised. Combine `β·KL(p_T ‖ M) + (1−β)·KL(p_S
‖ M)`, mask to completion tokens, reduce per token, temperature on both tensors before the softmax and
shared. On cost: two `log_softmax`, one `logsumexp` for `log M`, two `kl_div` reductions — roughly
three full-vocabulary passes against forward KL's one, but `log M` is one more `[B, T, V]` tensor of
exactly the width already materialized, so more arithmetic and no more memory than the forward-KL
default.

What this loss implements and does not. The general GKD object exposes the divergence as a tunable
knob, usually written with explicit `β = 0` and `β = 1` endpoint branches (forward and reverse KL)
plus the interior JSD so a trainer can sweep `β`. This baseline does *not* expose the sweep: it
hard-pins `beta_use = 0.5` and always computes the genuine interior JSD via the mixture, ignoring the
`beta` argument the signature passes. So it lands specifically the *symmetric Jensen-Shannon* point,
not the configurable family. The data axis stays the trainer's static `lmbda`, untouched; the only
change versus TAID is the divergence, from forward KL to symmetric JSD. The full body is in the answer.

Expectations against TAID and the cluster below. Symmetric JSD is half mode-seeking, so the prediction
is asymmetric. On MATH-500 I expect the *recovery* TAID's forward KL gave up: pulling the student off
pure mass-covering should let it commit to the teacher's reasoning modes, so MATH-500 should climb
back above the dagger/RS-KD level and above TAID's 0.280, a move large enough to clear the
`±10`-problem noise band. On GSM8K, where TAID's reachability fix already reached 0.4685 and short
arithmetic chains are more forgiving of a mass-covering match, I expect a smaller move — because the
divergence fix helps most exactly where mass-covering hurt most, and that was MATH-500, not GSM8K. If
instead GSM8K *jumped* and MATH-500 stayed flat, that would falsify the mass-covering diagnosis — it
would say mode-seeking helps the easy set, the opposite of the mechanism I derived. The open question
this leaves for the top of the progression is whether `β = 0.5` is far enough toward mode-seeking: if
half-JSD recovers MATH-500 but GSM8K only inches up, the natural next move is to push the divergence
all the way to the reverse-KL endpoint. AMC stays noise.
