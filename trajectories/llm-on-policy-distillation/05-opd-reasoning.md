Symmetric JSD did what its half-mode-seeking construction promised, and the residual it leaves is the
last lever. `baseline:gkd` reads GSM8K 0.4716, MATH-500 0.312, AMC 0.0781. On MATH-500 gkd solves 156
of 500 against taid's 140 and rs_kd's 146 — `+16` over taid, `+10` over rs_kd, a `+0.032` jump that
clears the `±2%` MATH-500 bar comfortably and is the highest so far. The recovery I
predicted is there and then some: it reverses taid's 0.280 collapse and clears the dagger/rs_kd
cluster decisively, the cleanest confirmation that the MATH-500 failures were a *divergence-direction*
problem, not a target or data problem. On GSM8K gkd solves 622 against taid's 618, `+4`, inside a
single standard error — the *smaller* move I expected on the set where short arithmetic chains forgive
a mass-covering match. So the half-JSD point bought a large MATH-500 recovery at essentially no GSM8K
cost, and that asymmetry is the diagnosis: the divergence direction is the binding axis, moving toward
mode-seeking is what pays, and the flat GSM8K move says half of the mode-seeking may be leaving value
on the table. So the natural top-of-progression move is to push the divergence *all the way* to the
reverse-KL endpoint and ask whether full mode-seeking on the student's own rollouts beats the balanced
compromise.

One number I read but refuse to over-read: gkd's final train loss came in at 0.0777, an order of
magnitude below everything below it. That is not "gkd fit its target ten times better." The symmetric
JSD is bounded above by `log 2 ≈ 0.693` on any single token, even one where student and teacher
disagree completely, and its typical per-token value is a fraction of that, so a converged JSD
averages a small number by *construction* — whereas a cross-entropy against a one-hot or a raw KL has
no such ceiling. The tiny train loss is a fact about the divergence's scale, not about how much math
the student learned; the accuracies, which rose, carry the signal.

Let me derive the endpoint from first principles rather than dialing `β → 1`, because the endpoint has
a cleaner story than "extreme JSD," and the story is what says it should win here. Two failures this
has been fighting all along. The data failure — training on a fixed distribution of prefixes while the
student walks its own at inference — is handled upstream by the `lmbda` mixing, which I priced at the
floor as turning the behavior-cloning `ε·T²` cascade into `ε·T`, and nothing since has needed to touch
it. The objective failure is what half-JSD only half-addressed: forward KL is mass-covering, forcing a
0.5B student to smear its limited capacity across a 7.6B teacher's long tail. The cleanest cure is not
a compromise on the JSD family but to flip the objective entirely to reverse KL
`KL(p_S ‖ p_T) = Σ_v p_S(v)·log(p_S(v)/p_T(v))`, weighted by the *student's* own probability: the
penalty is now large exactly where the student puts mass the teacher finds unlikely, so to minimize it
the student *withdraws* mass from anything the teacher dislikes — zero-forcing — and concentrates on
the modes the teacher genuinely favors. The half-JSD MATH-500 win is direct evidence that the more
mode-seeking I am, the better the hard set goes, so the endpoint is not an arbitrary extreme; it is
the pure form of the property that has been paying off all the way up.

Make it concrete with a toy I can compute by hand. Take a teacher with two separated modes and an
empty valley between, `p_T = [0.48, 0.02, 0.02, 0.48]` — a reasoning step with two coherent
continuations and incoherent middle ground. Give a capacity-limited student two options: hedge,
`s_cover = [0.25, 0.25, 0.25, 0.25]`, landing squarely in the empty valley; or commit,
`s_commit = [0.94, 0.02, 0.02, 0.02]`, snapping onto one mode. Forward KL `KL(p_T ‖ s)` gives `0.525`
for the hedge and `1.203` for the commit — forward *prefers the hedge*, because abandoning the token-3
mode where `p_T = 0.48` costs `0.48·log(0.48/0.02) ≈ 1.5` by itself, so it parks mass in the valley to
cover both. Reverse KL `KL(s ‖ p_T)` gives `0.937` for the hedge and `0.568` for the commit — reverse
*prefers the commit*, because now the `0.25`s in the empty valley cost `0.25·log(0.25/0.02) ≈ 0.63`
each and drive the student to withdraw and concentrate. Four numbers: forward parks the student in the
incoherent valley, reverse snaps it onto one mode and commits. On a long chain the valley is exactly
where reasoning goes to die, so "mode-seeking helps MATH-500" is the same statement as "get the
student out of the valley," and the pure reverse-KL endpoint is the strongest form of it.

Could I get the same mode-seeking more cautiously by pushing the JSD's `β` toward 1 — say `β = 0.9` —
keeping the bounded-on-disjoint-support safety? No, for two reasons pointing the same way. An interior
`β = 0.9` still forms the mixture `M = β·p_T + (1−β)·p_S` and computes *two* KLs against it; only at
the true corner `β → 1`, where `M → p_T` and the second term vanishes, does the loss collapse to the
single reverse KL whose immediate-reward gradient is exactly `−KL(p_S ‖ p_T)` computable as a
vocabulary sum — the property that lets me drop all the policy-gradient machinery. And the experiment I
want to run is precisely the *endpoint* hypothesis: half-JSD already told me mode-seeking pays, and the
sharp question is whether *full* mode-seeking beats the compromise; an interior `β` would blur exactly
that test. So I take the clean corner and accept that it gives up the JSD's automatic boundedness —
the one thing I will have to watch.

The moment I write reverse KL I have to answer why it is not trivially harder to optimize than forward
KL, because that objection is what makes people reach for JSD compromises. Forward KL `KL(p_T ‖ p_S)`
is an expectation under the *fixed* teacher, a clean supervised object. Reverse KL is
`E_{y ~ p_S}[log p_S − log p_T]`, an expectation *under the student's own distribution* — the thing I
differentiate sits inside an expectation over the distribution whose parameters I am changing. At the
sequence level that is an RL problem: the student samples trajectories `y ~ p_S(·|x)` and I maximize a
reward `r(y) = log p_T(y) − log p_S(y)`. Notice this *automatically* re-derives the on-policy data the
trainer already gives me — reverse KL and the on-policy loop are the same idea seen from two
directions, a reassuring sign I am at the right corner. But the naive estimator is the policy
gradient, and it is a mess: the reward-to-go `R_t = Σ_{t'≥t}(log p_T − log p_S)` is a high-variance sum
over the sampled future; a small student will *reward-hack* it, since degenerate repeated phrases can
earn high teacher probability locally while being garbage reasoning; and there is a length bias — if
the per-token reward is negative on average, a longer completion accumulates more negative reward, so
the gradient pushes toward *short or empty* answers, catastrophic for chain-of-thought math. Running
that needs variance baselines, teacher-mixed importance sampling, length normalization, PPO-style
clipping — machinery the single-loss edit surface cannot even express.

So look harder at *where* the mess is, because not all of the gradient is bad. Decompose the per-step
policy gradient into the immediate term and the long-horizon term. The immediate term's inner object is
`E_{y_t ~ p_S}[log(p_T(y_t)/p_S(y_t))] = −Σ_v p_S(v)·log(p_S(v)/p_T(v)) = −KL(p_S ‖ p_T)` at the
current prefix — the per-token reverse KL, evaluated *analytically over the whole vocabulary*. And the
setup hands me both full distributions at every position (the trainer forwards both models and gives me
both `[B, T, V]` tensors), so I compute this expectation *exactly* by summing over the vocabulary, no
sampling. Every pathology — the variance, the reward-hacking, the length bias — lives in the
*long-horizon* term carrying `R_{t+1}` and the REINFORCE score factor. The mess and the clean part
separate exactly along the immediate / long-horizon cut.

Do I need the long-horizon term? Its only job is credit assignment over a *sparse* reward — letting
token `t` be rewarded for good future tokens when the signal arrives later. But my reward is *dense*:
the teacher gives an informative signal at every single token. When the reward is dense, propagating
credit backward buys almost nothing and costs all the variance. So I drop the long-horizon term — a
discount `γ = 0` — and keep only the exact immediate conditional-KL gradient on the states the student
actually visits. What remains is a *supervised-style* loss: a differentiable function of the two logit
tensors at matching positions, evaluated on student-generated prefixes the trainer's `lmbda` already
supplies. Every stabilizer the policy gradient needed existed to tame the long-horizon term; with it
gone I need none — no baseline, no importance weights, no clipping. This is the `β = 1`, on-policy
corner of the same (divergence × data) family — maximally mode-seeking, maximally student-rollout —
and it is the corner the MATH-500 evidence has pointed at since the moving-target experiment split GSM8K up
and MATH-500 down.

The discount-zero choice removes the length bias by construction, not by tuning. With `γ = 0` there is
no sum over the future, so each token's loss is its own per-token reverse KL, independent of how many
tokens follow, and the completion length simply does not appear in any token's gradient. I make that
explicit in the reduction: `batchmean` divides the summed per-token KL by the number of valid
completion tokens, length-scale-free by construction, where a `sum` reduction would reintroduce a
length dependence through the back door.

The KL direction is the one trap; backwards would silently train forward KL — the *opposite* of the
whole point, the very mass-covering objective this progression climbed away from, with no error raised.
The framework's `kl_div(input=log_q, target=log_p, log_target=True)` computes
`KL(p ‖ q) = Σ_v p_target·(log p_target − input)`. For `KL(p_S ‖ p_T)` the *student* log-probs are the
target and the *teacher* log-probs the input — `kl_div(input=teacher_log_probs,
target=student_log_probs, log_target=True)`, summed over the vocabulary: the summand is
`p_S·(log p_S − log p_T)`, the reverse KL I want. Divide both tensors by the shared temperature first,
mask to completion tokens (`labels != −100`), reduce per token. The full body is in the answer.

What this task's loss does versus the method's broader form, so I land the literal edit and do not
over-claim. The full method is sometimes written with the policy-gradient / sampled-advantage
`(r_t − 1)·∇log p_S` realization for trainers that want the sampled estimator, and that recipe pins
`lmbda = 1.0`, always on-policy. This body is the *analytic vocabulary-sum* reverse KL, the preferred
form precisely when the teacher's full logits are available — and they are, every step — and it does
*not* change `lmbda`: it relies on the trainer's default `lmbda = 0.5` mixing and applies reverse KL
regardless of batch source. What makes this the reverse-KL endpoint is the *formulation*, not the
on-policy fraction; the only change versus the symmetric-JSD experiment is the divergence. That the analytic
form is also the *cheapest* — a single vocabulary sum against JSD's mixture-plus-two-KLs, no sampling,
no stabilizers — falls out of the dense-reward structure that let me drop the long-horizon term, and it
sharpens why the sampled realization is strictly worse *here*: a REINFORCE estimate replaces the exact
sum with a one-sample draw whose variance across a `~152k`-way vocabulary is exactly what forces the
baseline, the importance weights, and the clipping I would otherwise not need.

Expectations against gkd and the losses below. Full reverse KL is maximally mode-seeking, so the
prediction follows the logic half-JSD half-confirmed, pushed to the limit. On GSM8K I expect it to
clear gkd's 0.4716 by the *larger* margin that gkd's flat `+4`-problem move suggested was being left on
the table — committing fully to the teacher's preferred continuation on short, near-deterministic
arithmetic chains is where a confident single mode pays most, and the GSM8K counts have risen
monotonically every time I moved toward mode-seeking. MATH-500 is the genuine open question, and the
honest risk is the opposite ditch: reverse KL is zero-forcing, and zero-forcing taken to its unbounded
extreme can *over*-commit, collapsing the student onto too few of the teacher's modes. So on the hard
set I would not promise the continued climb — I would call it roughly level with gkd, plausibly a hair
under if the collapse bites. If that split appears — GSM8K clearly the best so far, MATH-500 level with
or a touch below gkd — this endpoint is the strongest baseline on the headline metric, and the residual
it leaves is precise: the mode-collapse risk of an *unbounded* reverse KL, whose gradient runs away
wherever the teacher assigns near-zero probability to what the student wrote. The next move would then
be a reverse-KL objective whose gradient stays bounded where the teacher mass vanishes — keeping the
mode-seeking commitment while removing the brittleness. AMC stays noise.
