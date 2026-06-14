Symmetric JSD did exactly what its half-mode-seeking construction promised, and the residual it leaves
is the last lever on the ladder. `baseline:gkd` reads GSM8K 0.4716, MATH-500 0.312, AMC 0.0781. The
MATH-500 recovery I predicted is there and then some: 0.312 is the *highest* MATH-500 of any rung so
far, reversing TAID's 0.280 collapse and clearing the dagger/RS-KD cluster (0.290/0.292) decisively —
which is the cleanest confirmation I have that the MATH-500 failures were a *divergence-direction*
problem, not a target or data problem. Pulling the student halfway off mass-covering let it commit to
the teacher's reasoning modes on the hard set. And GSM8K crept to 0.4716, a touch over TAID's 0.4685,
exactly the *smaller* move I expected on the set where mass-covering hurt least. So the half-JSD point
is the strongest interior-divergence rung, and it answers the question I posed: the divergence
direction is the binding axis, and moving toward mode-seeking is what pays. The open question it left
was whether `β = 0.5` is far enough. The MATH-500 recovery says mode-seeking helps; the modest GSM8K
move says half of it may be leaving value on the table. So the natural top-of-ladder move is to push
the divergence *all the way* to the reverse-KL endpoint and ask whether full mode-seeking on the
student's own rollouts beats the balanced compromise.

Let me derive that endpoint from first principles rather than just dialing `β → 1`, because the
endpoint has a cleaner story than "extreme JSD," and the story is what tells me it should win here.
Go back to the two failures the whole ladder has been fighting. The data failure — training on a fixed
distribution of prefixes while the student walks its own at inference — is already handled upstream by
the trainer's `lmbda` mixing, which trains on the student's own rollouts. The objective failure is the
one half-JSD only half-addressed: forward KL is mass-covering, and a 0.5B student cannot represent a
7.6B math model's full distribution, so any mass-covering pull smears its capacity. The cleanest cure
is not a compromise on the JSD family; it is to flip the objective entirely to reverse KL,
`KL(p_S ‖ p_T) = Σ_v p_S(v)·log(p_S(v)/p_T(v))`, weighted by the student's own probability, zero-
forcing: the student withdraws mass from anything the teacher dislikes and concentrates on the modes
the teacher genuinely favors. Under capacity mismatch, picking one teacher behavior and executing it
cleanly is precisely what I want, and the half-JSD MATH-500 win is direct evidence that the more
mode-seeking I am, the better the hard set goes.

Now, the moment I write reverse KL I have to be honest about why it is not trivially harder to optimize
than forward KL, because that objection is what makes people reach for JSD compromises in the first
place. Reverse KL is `E_{p_S}[log p_S − log p_T]`, an expectation *under the student's own
distribution* — the thing I am differentiating sits inside an expectation over the distribution whose
parameters I am changing. At the sequence level that is an RL problem: the student is a policy sampling
trajectories `y ~ p_S(·|x)`, minimize the sequence reverse KL. Notice this *automatically* re-derives
the on-policy data the trainer already gives me — the expectation is over the student's own rollouts —
so reverse KL and the on-policy loop are the same idea seen from two directions. But the naive
estimator is the policy gradient, and it is a mess for the reasons I can name in advance: the
reward-to-go `R_t = Σ_{t'≥t} (log p_T − log p_S)` is a high-variance sum over the sampled future;
worse, a small student will *reward-hack* it (degenerate repeated phrases get high teacher probability
locally while being garbage reasoning), and there is a length bias baked into `R_t` that pushes toward
short or empty answers. Run that and I would need variance baselines, teacher-mixed sampling with
importance weights, length normalization, PPO clipping — a pile of machinery that drags me away from
the stable supervised loop the ladder has lived in.

So let me look harder at *where* the mess is, because not all of the gradient is bad. Decompose the
per-step gradient into the immediate term and the long-horizon term. The immediate term is the
gradient of the *expected immediate reward* at each step, and its inner object is
`E_{y_t~p_S}[log(p_T(y_t)/p_S(y_t))] = −Σ_v p_S(v)·log(p_S(v)/p_T(v)) = −KL(p_S(·|y_{<t}) ‖ p_T(·|y_{<t}))`
— the per-token reverse KL, evaluated *analytically over the whole vocabulary* at the current prefix.
I have both the student and teacher full distributions at every position (the trainer hands me both
logit tensors), so I can compute this expectation exactly by summing over the vocabulary, with no
sampling noise at all. Every pathology — the variance, the reward-hacking via sampled completions, the
reward-to-go length bias — lives in the *long-horizon* term, the one with `R_{t+1}` and the REINFORCE
score factor and the sampled future. The immediate term is clean.

Do I need the long-horizon term? Its only job is credit assignment over a sparse reward — letting token
`t` be rewarded for good *future* tokens it enabled. But my reward is *not* sparse: the teacher gives a
dense, informative signal at *every single token*. I do not have to wait for the future to learn
whether token `t` was good; the teacher tells me now, at token `t`, by how much probability it assigns
to what the student did. The entire reason to tolerate the variance of long-horizon credit assignment
is sparse reward, and I do not have sparse reward. So I drop the long-horizon term — set its weight to
zero, a discount factor of zero — and keep only the exact immediate conditional-KL gradient on the
states the student actually visits. What remains is startlingly simple: at each on-policy position,
compute the per-token reverse KL between student and teacher analytically over the vocabulary, sum over
the completion tokens, backprop. It is structurally a *supervised* loss — a differentiable function of
the two logit tensors at matching positions — that just happens to be evaluated on student-generated
prefixes (which the trainer's `lmbda` already supplies). Every stabilizer the policy gradient needed
existed to tame the long-horizon term, and with it gone I need none of them. This is the `β = 1`,
on-policy corner of the same (divergence × data) family the rest of the ladder lives in — maximally
mode-seeking, maximally student-rollout — and it is the corner the MATH-500 evidence has been pointing
at since TAID.

The KL direction is the one trap, and getting it backwards would silently train forward KL — the
*opposite* of the whole point, the very mass-covering objective the ladder has been climbing away from.
The framework's `kl_div(input=log_q, target=log_p, log_target=True)` computes `KL(p ‖ q)`: it sums
`Σ_v p_(target)·(log p_target − input)`. I want `KL(p_S ‖ p_T) = Σ_v p_S·(log p_S − log p_T)`, so the
*student* log-probs are the target and the *teacher* log-probs are the input — the call is
`kl_div(input=teacher_log_probs, target=student_log_probs, log_target=True)`, summed over the
vocabulary. The off-the-shelf default with the arguments swapped would compute forward KL; the two
differ only by which tensor is which, which is exactly why it is worth stating out loud. Divide both
logit tensors by the shared temperature first (symmetric, so the KL measures behavior, not a sharpness
mismatch), mask to completion tokens (`labels != -100`), reduce per token — `batchmean` is the per-
token mean over kept tokens, length-scale-free, which matters because the discount-zero design already
killed the reward-to-go length bias and the per-token reduction makes that explicit. The full scaffold
body is in the answer.

One note on what this task's loss does versus the method's broader form, so I land the literal edit.
The full method is sometimes written with the policy-gradient / sampled-advantage realization (the
`(r_t − 1)·∇log p_S` score-function form) for trainers that want the sampled estimator, and it pins
`lmbda = 1.0` (always on-policy) as part of the recipe. This task's loss body is the *analytic
vocabulary-sum* reverse KL — the preferred form when the teacher's full logits are available, which
they are — and it does *not* change `lmbda`: it relies on the trainer's default `lmbda = 0.5` mixing
and applies reverse KL regardless of whether a given batch came from the student or the dataset. So
what makes this OPD is the *reverse-KL formulation*, not a change to the on-policy fraction; the data
axis stays the trainer's static mixing, exactly as on every prior rung, and the only thing this rung
changes versus GKD is the divergence, from symmetric JSD to the pure reverse-KL endpoint.

Falsifiable expectations against GKD and the rungs below. Full reverse KL is maximally mode-seeking, so
the prediction follows the same logic that half-JSD half-confirmed, pushed to the limit. On GSM8K I
expect it to clear GKD's 0.4716 by the *larger* margin that GKD's modest GSM8K move suggested was being
left on the table — into the high-0.48s — because committing fully to the teacher's preferred
continuation on short, near-deterministic arithmetic chains is where a confident single mode pays most.
MATH-500 is the genuine open question, and the honest risk is the mode-collapse ditch: full reverse KL
could overshoot half-JSD's balanced compromise and *drop* a touch from GKD's 0.312 if the student
collapses onto too few of the teacher's reasoning modes on the hard set — so I would call it roughly
level with GKD, low-0.31s, plausibly a hair under. If that is the pattern — GSM8K clearly the best on
the ladder, MATH-500 level with GKD — then OPD is the strongest baseline overall on the headline
metric, and the residual it leaves is precisely the mode-collapse risk of an *unbounded* reverse KL:
the next thing to try would be a reverse-KL objective whose gradient is bounded where the teacher mass
vanishes, so the student gets the mode-seeking commitment without the brittleness. AMC stays noise; I
will not read it.
