Symmetric JSD did exactly what its half-mode-seeking construction promised, and the residual it leaves
is the last lever on the ladder. `baseline:gkd` reads GSM8K 0.4716, MATH-500 0.312, AMC 0.0781. Read
the MATH-500 number against the two rungs it was supposed to fix: gkd solves 156 of 500 there, against
taid's 140 and rs_kd's 146 — `+16` problems over taid, `+10` over rs_kd, a `+0.032` jump that clears
the `±2%` MATH-500 standard-error bar comfortably and is the *highest* MATH-500 of any rung so far. The
recovery I predicted is there and then some: it reverses taid's 0.280 collapse and clears the
dagger/rs_kd cluster (0.290/0.292) decisively, which is the cleanest confirmation I have that the
MATH-500 failures were a *divergence-direction* problem, not a target or data problem. Pulling the
student halfway off mass-covering let it commit to the teacher's reasoning modes on the hard set. Now
read GSM8K the same way: gkd solves 622 against taid's 618, `+4` problems, `+0.003` — inside a single
standard error on GSM8K, exactly the *smaller* move I expected on the set where short arithmetic chains
forgive a mass-covering match. So the half-JSD point bought a large MATH-500 recovery at essentially no
GSM8K cost, and that asymmetry is the whole diagnosis: the divergence direction is the binding axis,
moving toward mode-seeking is what pays, and the fact that GSM8K barely moved says half of the
mode-seeking may be leaving value on the table. The MATH-500 win says mode-seeking helps; the flat
GSM8K move says I have not spent all of it. So the natural top-of-ladder move is to push the divergence
*all the way* to the reverse-KL endpoint and ask whether full mode-seeking on the student's own
rollouts beats the balanced compromise.

One number I will read but refuse to over-read: gkd's final training loss came in at 0.0777, against
taid's 0.363, rs_kd's 0.342, and dagger's 0.507 — an order of magnitude below everything below it. It
would be a mistake to call that "gkd fit its target ten times better." The rule I set two rungs ago
holds — these are different objectives on different supports, not commensurable — but this rung's
comparison makes it sharper than a rule, because I can say exactly *why* the number is small. The
symmetric Jensen-Shannon divergence is bounded above by `log 2 ≈ 0.693` on any single token, even one
where the student and teacher disagree completely, and its typical per-token value is a fraction of
that, so a converged JSD averages a small number by *construction* — whereas a cross-entropy against a
one-hot (dagger) or a raw KL (rs_kd, taid) has no such ceiling and sits higher for the same quality of
fit. The tiny train loss is a fact about the divergence's scale, not about how much math the student
learned; the accuracies, which rose, carry the signal, and the train-loss column stays a within-loss
convergence diagnostic exactly as it has been.

Let me derive that endpoint from first principles rather than just dialing `β → 1`, because the
endpoint has a cleaner story than "extreme JSD," and the story is what tells me it should win here
rather than merely that it is the corner I have not tried. Go back to the two failures the whole ladder
has been fighting. The data failure — training on a fixed distribution of prefixes while the student
walks its own at inference — is already handled upstream by the trainer's `lmbda` mixing, which trains
on the student's own rollouts; I derived at the floor that this turns the behavior-cloning `ε·T²`
cascade into `ε·T`, and nothing since has needed to touch it. The objective failure is the one that
half-JSD only half-addressed: forward KL is mass-covering, `KL(p_T ‖ p_S) = Σ_v p_T(v)·log(p_T/p_S)`
weights every token by the *teacher's* mass and so forces the student to put probability everywhere the
teacher does, and a 0.5B student cannot represent a 7.6B math model's full distribution, so any
mass-covering pull smears its limited capacity across the teacher's long tail. The cleanest cure is not
a compromise on the JSD family; it is to flip the objective entirely to reverse KL,
`KL(p_S ‖ p_T) = Σ_v p_S(v)·log(p_S(v)/p_T(v))`, weighted by the *student's* own probability. That
reweighting is the whole point: the penalty is now large exactly where the student puts mass the
teacher finds unlikely, so to minimize it the student *withdraws* mass from anything the teacher
dislikes — zero-forcing — and concentrates on the modes the teacher genuinely favors. Under capacity
mismatch, picking one teacher behavior and executing it cleanly is precisely what I want, and the
half-JSD MATH-500 win is direct evidence that the more mode-seeking I am, the better the hard set goes.
So the endpoint is not an arbitrary extreme; it is the pure form of the exact property that has been
paying off all the way up the ladder.

Let me make the mass-covering-versus-mode-seeking distinction concrete with a toy I can compute by
hand, because I keep asserting it and I want the numbers to commit to one side. Take a teacher with two
separated modes and an empty valley between them, `p_T = [0.48, 0.02, 0.02, 0.48]` — mass on tokens 0
and 3, near-nothing on 1 and 2, a stand-in for a reasoning step with two coherent continuations and
incoherent middle ground. Give a capacity-limited student two options: hedge, `s_cover =
[0.25, 0.25, 0.25, 0.25]`, spreading mass and landing squarely in the empty valley; or commit,
`s_commit = [0.94, 0.02, 0.02, 0.02]`, snapping onto one mode. Now score them. Forward KL
`KL(p_T ‖ s)` gives `0.525` for the hedge and `1.203` for the commit — forward *prefers the hedge*,
because abandoning the token-3 mode where `p_T = 0.48` costs `0.48·log(0.48/0.02) ≈ 1.5` all by itself,
so forward KL will not let the student drop a teacher mode and it parks mass in the valley to cover
both. Reverse KL `KL(s ‖ p_T)` gives `0.937` for the hedge and `0.568` for the commit — reverse
*prefers the commit*, because now the penalty is on student mass sitting where the teacher is thin, so
the `0.25`s in the empty valley cost `0.25·log(0.25/0.02) ≈ 0.63` each and drive the student to
withdraw from the valley and concentrate. That is the whole asymmetry in four numbers: forward KL parks
the student in the incoherent valley to cover both modes, reverse KL snaps it onto one and commits. On
a long competition chain the valley is exactly where reasoning goes to die — a token sampled from the
hedge is a blend of two incompatible continuations — so the MATH-500 evidence that mode-seeking helps
is the same statement as "get the student out of the valley," and the pure reverse-KL endpoint is the
strongest form of that.

Before I commit to the pure endpoint I should ask whether I could get the same mode-seeking more
cautiously by pushing the JSD's `β` up toward 1 — say `β = 0.9` — rather than jumping clean off the
family, because that would keep the bounded-on-disjoint-support safety the JSD rung bought. I reject
it, for two reasons that both point the same way. First, an interior `β = 0.9` still forms the mixture
`M = β·p_T + (1−β)·p_S` and computes *two* KLs against it, so it never becomes the clean analytic
object: it is only at the true corner `β → 1`, where `M → p_T` and the second JSD term vanishes, that
the loss collapses to the single reverse KL whose immediate-reward gradient is exactly `−KL(p_S ‖ p_T)`
computable as a vocabulary sum — the property that lets me drop all the policy-gradient machinery. An
interior `β` gives me neither the clean gradient identity nor a nameable "this is the endpoint"
experiment. Second, the experiment I actually want to run is precisely the *endpoint* hypothesis: the
half-JSD rung already sits at the balanced interior and told me mode-seeking pays; the sharp question
it left is whether *full* mode-seeking beats the compromise, and an interior `β = 0.9` would blur
exactly that test the way a ten-epoch warmup blurs a no-warmup test. So I take the clean corner, and I
accept that it gives up the JSD's automatic boundedness — which is the one thing I will have to watch,
because an unbounded reverse KL is only as safe as the tokens it is evaluated on.

Now, the moment I write reverse KL I have to be honest about why it is not trivially harder to optimize
than forward KL, because that objection is what makes people reach for JSD compromises in the first
place, and if I cannot answer it I have no business at this endpoint. Forward KL `KL(p_T ‖ p_S)` is an
expectation under the *fixed* teacher, so its gradient in the student is a clean supervised object.
Reverse KL is `E_{y ~ p_S}[log p_S − log p_T]`, an expectation *under the student's own distribution* —
the thing I am differentiating sits inside an expectation over the distribution whose parameters I am
changing. At the sequence level that is an RL problem: the student is a policy sampling trajectories
`y ~ p_S(·|x)`, and I am minimizing the sequence reverse KL, i.e. maximizing a reward
`r(y) = log p_T(y) − log p_S(y)`. Notice this *automatically* re-derives the on-policy data the trainer
already gives me — the expectation is over the student's own rollouts — so reverse KL and the on-policy
loop are the same idea seen from two directions, which is a reassuring sign I am at the right corner and
not fighting the substrate. But the naive estimator is the policy gradient, and it is a mess for
reasons I can name in advance. The reward-to-go `R_t = Σ_{t'≥t}(log p_T − log p_S)` is a high-variance
sum over the sampled future; worse, a small student will *reward-hack* it, because degenerate repeated
phrases can earn high teacher probability locally while being garbage reasoning, and there is a length
bias baked into `R_t` — if the per-token reward is negative on average (the student is usually a little
off the teacher), then a longer completion accumulates more negative reward, so the gradient pushes
toward *short or empty* answers, which is catastrophic for chain-of-thought math. Run that and I would
need variance baselines, teacher-mixed sampling with importance weights, length normalization, PPO-style
clipping — a pile of machinery that drags me away from the stable supervised loop the whole ladder has
lived in, and that this single loss-body edit surface cannot even express.

So let me look harder at *where* the mess is, because not all of the gradient is bad and I should not
throw out a clean term with the dirty one. Decompose the per-step policy gradient into the immediate
term and the long-horizon term. The immediate term is the gradient of the *expected immediate reward*
at each step, and its inner object is
`E_{y_t ~ p_S}[log(p_T(y_t)/p_S(y_t))] = Σ_v p_S(v)·log(p_T(v)/p_S(v)) = −Σ_v p_S(v)·log(p_S(v)/p_T(v))
= −KL(p_S(·|y_{<t}) ‖ p_T(·|y_{<t}))` — the per-token reverse KL, evaluated *analytically over the whole
vocabulary* at the current prefix. And here is the gift the setup hands me: I have both the student and
teacher full distributions at every position, because the trainer forwards both models and gives me both
logit tensors `[B, T, V]`, so I can compute this expectation *exactly* by summing over the vocabulary,
with no sampling estimate at all. Every pathology I listed — the variance, the reward-hacking via
sampled completions, the reward-to-go length bias — lives in the *long-horizon* term, the one carrying
`R_{t+1}`, the REINFORCE score factor, and the sampled future. The immediate term, computed as a
vocabulary sum, has none of them: it is a deterministic, differentiable function of the two logit
tensors at matching positions. The mess and the clean part separate exactly along the immediate /
long-horizon cut.

Do I need the long-horizon term? Its only job is credit assignment over a *sparse* reward — letting
token `t` be rewarded for good *future* tokens it enabled, when the signal only arrives later. But my
reward is *not* sparse: the teacher gives a dense, informative signal at *every single token*, telling
me right now, at position `t`, how much probability it assigns to what the student did there. The entire
reason to tolerate the variance of long-horizon credit assignment is that the reward is delayed; when
the reward is dense at every token, propagating credit backward from the future buys almost nothing and
costs all the variance. So I drop the long-horizon term — set its weight to zero, a discount factor of
`γ = 0` — and keep only the exact immediate conditional-KL gradient on the states the student actually
visits. What remains is startlingly simple: at each on-policy position, compute the per-token reverse KL
between student and teacher analytically over the vocabulary, sum over the completion tokens, backprop.
It is structurally a *supervised* loss — a differentiable function of the two logit tensors at matching
positions — that just happens to be evaluated on student-generated prefixes, which the trainer's `lmbda`
already supplies. Every stabilizer the policy gradient needed existed to tame the long-horizon term, and
with it gone I need none of them: no baseline, no importance weights, no clipping. This is the `β = 1`,
on-policy corner of the same (divergence × data) family the rest of the ladder lives in — maximally
mode-seeking, maximally student-rollout — and it is the corner the MATH-500 evidence has been pointing
at since the moving-target rung split GSM8K up and MATH-500 down.

Let me close the loop between the toy and the object I just defined, because it is easy to define a
loss and never check it computes what the derivation says. The discount-zero loss is, per token, the
analytic vocabulary sum `Σ_v p_S(v)·(log p_S(v) − log p_T(v))` — which is exactly `KL(p_S ‖ p_T)`, the
same quantity I hand-computed on the toy. On `s_commit = [0.94, 0.02, 0.02, 0.02]` against
`p_T = [0.48, 0.02, 0.02, 0.48]` that sum is `0.94·log(0.94/0.48) + 0.02·log(0.02/0.02) +
0.02·log(0.02/0.02) + 0.02·log(0.02/0.48) = 0.94·0.672 + 0 + 0 + 0.02·(−3.178) = 0.632 − 0.064 =
0.568`, the number I already got for the committing student. So the loss I will actually backprop is
*literally* the reverse KL that scored the committing student below the hedging one — the analytic
form and the mode-seeking preference are the same object, not two facts I am hoping line up. The body
computes the divergence the derivation promised.

Let me sanity-check the discount-zero choice against the length bias specifically, because that is the
pathology I most feared and I want to confirm the fix actually removes it rather than hiding it. The
length bias came from `R_t` summing a per-token reward over the whole future, so the *number* of future
tokens entered the objective and the gradient could trade accuracy for brevity. With `γ = 0` there is no
sum over the future at all — each token's loss is its own per-token reverse KL, independent of how many
tokens follow — so the completion length simply does not appear in any token's gradient. And I make that
explicit in the reduction: `batchmean` divides the summed per-token KL by the number of valid completion
tokens, a per-token mean, which is length-scale-free by construction. A `sum` reduction would have
re-introduced a length dependence through the back door (longer completions contributing more total
loss); the per-token mean does not. So the discount-zero design and the per-token reduction agree, and
the length bias is gone by construction rather than by tuning — I did not have to add a
length-normalization hack, because refusing the long-horizon term already removed the term that carried
the bias.

The KL direction is the one trap, and getting it backwards would silently train forward KL — the
*opposite* of the whole point, the very mass-covering objective the ladder has been climbing away from,
with no error raised to warn me. The framework's `kl_div(input=log_q, target=log_p, log_target=True)`
computes `KL(p ‖ q) = Σ_v p_target·(log p_target − input)`: it treats the *target* as the distribution
the KL is from and the *input* as the log-denominator. I want `KL(p_S ‖ p_T) = Σ_v p_S·(log p_S −
log p_T)`, so the *student* log-probs are the target and the *teacher* log-probs are the input — the call
is `kl_div(input=teacher_log_probs, target=student_log_probs, log_target=True)`, summed over the
vocabulary. The off-the-shelf default with the arguments swapped would compute forward KL; the two differ
only by which tensor is which, which is exactly why it is worth stating out loud and checking once by
hand: with `input = teacher` and `target = student`, the summand is `p_S·(log p_S − log p_T)`, which is
the reverse KL I want. Divide both logit tensors by the shared temperature first (symmetric, so the KL
measures a behavior mismatch and not a sharpness mismatch), mask to completion tokens
(`labels != −100`), reduce per token. The full scaffold body is in the answer.

One note on what this task's loss does versus the method's broader form, so I land the literal edit and
do not over-claim what is in scope. The full method is sometimes written with the policy-gradient /
sampled-advantage realization — the `(r_t − 1)·∇log p_S` score-function form — for trainers that want the
sampled estimator, and that recipe pins `lmbda = 1.0`, always on-policy, as part of its definition. This
task's loss body is the *analytic vocabulary-sum* reverse KL, which is the preferred form precisely when
the teacher's full logits are available, and they are — the trainer hands me the whole `[B, T, V]`
teacher tensor every step. And it does *not* change `lmbda`: it relies on the trainer's default
`lmbda = 0.5` mixing and applies reverse KL regardless of whether a given batch came from the student or
the dataset, exactly as every prior rung consumed that mixing. So what makes this the reverse-KL endpoint
is the *reverse-KL formulation*, not a change to the on-policy fraction; the data axis stays the
trainer's static mixing, and the only thing this rung changes versus the symmetric-JSD rung is the
divergence, from the half-mode-seeking interior to the pure reverse-KL corner. That the analytic form
also happens to be the cheap, stable one — a single vocabulary-sum, no sampling, no stabilizers — is a
bonus that comes from the dense-reward structure, not a compromise I am making.

A cost note, since it cuts the other way from the last rung. Where symmetric JSD needed a mixture
`logsumexp` and *two* `kl_div` reductions per token, the analytic reverse KL is a *single* vocabulary
sum over the two log-prob tensors — one `kl_div` call — so OPD is actually cheaper than the rung below
it, back to the forward-KL default's one-pass cost at the same `[B, T, V]` memory. That the
strongest-mode-seeking option is also the cheapest is not a coincidence I engineered; it falls out of
the dense-reward structure that let me drop the long-horizon term. And it sharpens why the sampled
realization is not merely inelegant but strictly worse *here*: a REINFORCE estimate replaces the exact
sum `Σ_v p_S(v)·log(p_S(v)/p_T(v))` with a one-sample draw `log(p_S(y_t)/p_T(y_t))` at the sampled
token, whose variance across a `~152k`-way vocabulary is exactly what forces the baseline, the
importance weights, and the clipping. When I already hold both full distributions, paying that variance
to estimate a quantity available in closed form is a bad trade before I even count the stabilizers it
drags in.

Falsifiable expectations against gkd and the rungs below, and because this is a prediction I have to
make it before the number exists. Full reverse KL is maximally mode-seeking, so the prediction follows
the same logic that half-JSD half-confirmed, pushed to the limit. On GSM8K I expect it to clear gkd's
0.4716 by the *larger* margin that gkd's flat `+4`-problem GSM8K move suggested was being left on the
table — into the high-0.48s — because committing fully to the teacher's preferred continuation on short,
near-deterministic arithmetic chains is where a confident single mode pays most, and the ladder's GSM8K
counts have risen monotonically every time I moved toward mode-seeking. MATH-500 is the genuine open
question, and the honest risk is the opposite ditch: reverse KL is zero-forcing, and zero-forcing taken
to its unbounded extreme can *over*-commit, collapsing the student onto too few of the teacher's
reasoning modes. So on the hard set I would not promise the continued climb; I would call it roughly
level with gkd, low-0.31s, plausibly a hair under if the collapse bites. If that split is what appears —
GSM8K clearly the best on the ladder, MATH-500 level with or a touch below gkd — then this endpoint is
the strongest baseline overall on the headline metric, and the residual it leaves is precise: the
mode-collapse risk of an *unbounded* reverse KL, whose gradient runs away wherever the teacher assigns
near-zero probability to what the student wrote. The next thing to try would then be a reverse-KL
objective whose gradient stays bounded where the teacher mass vanishes — keeping the mode-seeking
commitment that won GSM8K while removing the brittleness that capped MATH-500. AMC stays noise; I will
not read it.
