The grpo run came back and it told me where the std is hurting me, in numbers. On the easy split it
held — GSM8K landed at 0.4668, right where a working critic-free estimator should be — so the loop is
learning and the group-mean baseline is doing its job. But on the harder splits the accuracy is exactly
where I feared the per-group std would leave it: MATH-500 at 0.2973 and AMC at 0.0934, with an
aggregate `score_mean` of −0.5166, the *worst* of the three estimators I have to compare against. That
pattern is the tell. The std distortion I flagged at step 1 bites hardest precisely on the prompts
whose reward spread is large or whose 16-sample groups are near-unanimous — and those are
disproportionately the MATH and AMC prompts, not the GSM8K ones. So the floor is not failing to learn;
it is leaving accuracy on the table on the harder benchmarks, and the suspect is the one term I was
never sure about. Time to interrogate it properly rather than on a hunch.

First let me read the numbers more carefully than "GSM8K held, the rest sagged," because the table
carries information I can use. GSM8K accuracy is 0.4668 and `gsm8k_reward_mean` is also 0.4668 — not
close, identical. That pins something down: the verifier reward is the correctness indicator itself, so
the mean reward on a benchmark equals the fraction solved, and `r_i ∈ {0, 1}`. Good — my whole
group-statistics analysis lives on 0/1 rewards, and now I know that for certain rather than assuming
it. Next, the three accuracies form a steep ladder: 0.4668, 0.2973, 0.0934, whose unweighted mean is
`(0.4668 + 0.2973 + 0.0934)/3 = 0.2858`. The reported `score_mean` of −0.5166 is a normalized aggregate
(it also folds in the hidden AIME split), so I will not conflate the two, but the ladder itself is the
diagnostic: GSM8K is `1.57×` MATH-500 and MATH-500 is `3.18×` AMC. AMC at 9.3% is where a 0.5B model is
essentially at the floor of competence, so it is the most sensitive probe of whether the training update
is spending its gradient on the right prompts. That is the split I most want to move, and it is the one
the std should have hurt most.

Let me write down exactly what the grpo estimator does and look at it as if I had never seen it. For a
prompt I draw 16 responses, score them `R = {R_1,…,R_G}`, and set the advantage of every token of
response `i` to `(R_i − mean(R)) / (std(R) + ε)`, broadcast over its tokens. The group mean is doing
the job a value network would do — my cheap baseline, no critic to train, which is the whole reason I am
here. I believe in the baseline; GSM8K confirms it. But the `std(R)` in the denominator I have been
treating with suspicion since step 1, and I want to ask, very literally: what does that term do to the
gradient on a single token, and is there any version of it I should keep?

Before I delete anything, let me lay out the honest options for that denominator, because "it looks
bad, remove it" is not a derivation. One option is to keep the per-group std but floor it more
aggressively — clip it from below at, say, 0.25 so near-unanimous groups cannot blow up. Another is to
replace the per-group std with a single scale computed over a much larger pool. Another is to delete it
outright and divide by nothing. Walk the flooring option, since it is the least disruptive: flooring the
std caps the `3.75`-style blowups from near-unanimous groups, so it treats the *brittleness*. But it
does nothing about the *scope* — every prompt is still divided by its own number, so the difficulty
reweighting I computed at step 1 survives intact; a balanced group still gets divided by ~0.52 and a
near-unanimous one by the floor 0.25, and the 4× tilt toward uninformative prompts barely moves. So
flooring is a patch on the symptom that leaves the disease. The larger-pool option is more principled,
but it is a real redesign and I should not reach for it before I have established that the *clean*
thing — deleting the term — lands me somewhere defensible. If plain deletion turns out to be a
principled estimator in its own right, that is the move to make first, and any scale question becomes a
separate, later question.

So does deletion leave me with an estimator I can defend, or just "grpo minus a term"? The reading I
carry around for the std is "whitening advantages to zero mean and unit variance stabilizes training."
That is true — but the scope is where it goes wrong. When you whiten advantages across an entire batch,
you divide every advantage by one global number, the batch std, and that is just a uniform rescale of
the whole gradient — it folds into the learning rate and changes nothing about the relative weighting of
examples. But grpo computes the std *per prompt, per group*. Each prompt's centered reward is divided by
*that prompt's* std. So different prompts are divided by different numbers, and that does change their
relative weight in the update. Make it concrete the way I did at step 1: pull out the un-normalized
centered score `Ã_i = R_i − mean(R)` and read grpo as a reweighting of it, `Ã_i / std(R)`. A prompt
whose 16 responses are almost all correct has tiny std; one almost all wrong also has tiny std; a prompt
with a genuine mix — the informative ones — has large std. Dividing by std multiplies *up* the update
weight of the too-easy and too-hard prompts and tamps *down* the mixed-outcome ones. And this is exactly
why the *harder* splits suffered most in the grpo numbers: MATH and AMC are where the mix of solved and
unsolved is real, so those are precisely the prompts whose large std gets them tamped down, while the
trivially-easy and trivially-hard prompts (more common on GSM8K) get their weight inflated. The −0.5166
score and the depressed 0.2973 / 0.0934 are the difficulty bias made visible.

Now I want to be sure this term is not supposed to be there, so I derive the gradient from scratch and
watch whether a std ever appears. I am maximizing `J(π_θ) = E_q E_{o~π_θ}[R(q,o)]`. (I drop any
KL-to-reference term from the advantage reasoning — with a rule-based verifier there is no reward-model
distribution I must stay near in the *advantage*, and in any case the KL-loss setting is fixed outside
my edit.) The Monte-Carlo policy gradient is `∇J = E[ ∇log π_θ(o|q) · R(q,o) ]`. The log-prob
factorizes over tokens, so `∇J = E[ Σ_t ∇log π_θ(o_t|·) · R(q,o) ]`. Now the standard sharpening: a
token at position `t` cannot influence rewards before `t`, so I may replace `R(q,o)` multiplying the
`t`-th score by the reward-to-go `Σ_{t'≥t} r(q,o_{≤t'})` without changing the expectation. Then the
baseline: subtract `B(q,o_{<t})` to cut variance; as long as `B` does not depend on the action `o_t`,
`E_{o_t}[ ∇log π_θ(o_t|·) B ] = B · ∇_θ Σ_{o_t} π_θ = B · ∇_θ 1 = 0`, so it is unbiased. The thing in
parentheses, `reward-to-go − B`, is the advantage; the textbook-best `B` is the expected reward-to-go,
the state value — exactly the critic I am avoiding.

Now use the structure of *my* reward. It is outcome-level: zero everywhere except the last token. So
the reward-to-go from *any* position `t` is the same — the whole-trajectory return `R(q,o)`, for every
`t`. That collapses everything: the advantage is one scalar per response, broadcast across its tokens.
There is no per-token credit because there is no per-token reward. The entire estimator reduces to "pick
a baseline `B`, compute `R_i − B`, broadcast." Everything is in the choice of `B`. I do not have a value
network; I have 16 samples from the same prompt, whose mean is a Monte-Carlo estimate of the prompt's
expected return.

Since everything now rides on `B`, let me make sure the mean is the right choice and not just the
obvious one, by walking the alternatives on 0/1 rewards. A median baseline sounds robust, but with
binary rewards the group median is itself 0 or 1 — for a `k < 8` group it is 0, for `k > 8` it is 1 — so
median-centering would hand out advantages in `{0, 1}` or `{−1, 0}` and throw away the graded
fraction-correct information entirely; a prompt solved 7/16 and one solved 1/16 would both center
against 0 and look identical. Useless. A fixed constant baseline, say `B = 0.5`, is worse: it does not
adapt to the prompt at all, so on a hard prompt with `mean = 0.1` every response — *including the one
correct solution* — sits below 0.5 and gets a negative advantage, and the update would push the policy
*away* from its own rare correct answer. That is the difficulty-as-quality error again, now entering
through the baseline. Both alternatives fail for the same reason: they discard the per-prompt, graded
quantity the mean captures. So the mean stays; it is the load-bearing part, and it is the std, not the
baseline, that I am removing. Center by that mean: `Ã_i = R_i − mean(R)`, constant in `t`. No std has
appeared. No length term has appeared. The derivation asks for a *centered* return and stops.

But I must check this centered baseline before I call it unbiased: the mean includes `R_i` itself, so it
is not action-independent in the literal proof sense. The clean multi-sample baseline is leave-one-out,
`B_i = (1/(G−1)) Σ_{j≠i} R_j`, which by construction does not depend on `o_i`. How far is the
group-mean-centered advantage from leave-one-out? Scale the centered advantage by `G/(G−1)`:
`(G/(G−1))(R_i − (1/G)Σ_j R_j) = R_i − (1/(G−1))Σ_{j≠i} R_j`, which *is* the leave-one-out advantage. So
`(G/(G−1)) · (R_i − mean(R)) = R_i − mean_{j≠i}(R_j)` exactly: the mean-only centered advantage is the
unbiased leave-one-out policy gradient up to the global constant `(G−1)/G` that folds into the learning
rate.

Let me not take that on faith; let me trace it on a concrete group. Balanced prompt, `k = 8` correct of
16, so `mean = 0.5`, `G/(G−1) = 16/15 = 1.0667`. Take a *correct* response, `R_i = 1`. Mean-only gives
`1 − 0.5 = 0.5`. Leave-one-out: the other 15 responses hold 7 correct, so `B_i = 7/15 = 0.4667` and the
advantage is `1 − 0.4667 = 0.5333`. And `0.5 × 16/15 = 0.5333` — they match. Take a *wrong* response,
`R_i = 0`. Mean-only gives `−0.5`; leave-one-out sees 8 correct among the other 15, `B_i = 8/15 =
0.5333`, advantage `0 − 0.5333 = −0.5333 = −0.5 × 16/15`. They match again, in both sign and magnitude.
So deleting the std does not leave me with a biased fragment; it leaves me holding the unbiased RLOO
estimator, uniformly rescaled by `15/16` — a factor the learning rate cannot tell apart from itself.
That is the confirmation I wanted, and it converts "delete the std because it looks bad" into "delete
the std because the gradient never asked for it."

There is a second, sharper way to see why deletion is not just harmless but *corrective*, and it is the
squared-mass view from step 1 read forward. Under grpo's z-score every non-unanimous group is forced to
total advantage mass `Σ_i z_i² = G−1 = 15`, identical across prompts. Under mean-only centering the mass
is `Σ_i (R_i − mean)² = (G−1)·samplevar = 16·p(1−p)`: `4` for a balanced group, `3` for `k = 4`, and
`0.9375` for a near-unanimous `k = 1` or `k = 15`, and `0` for a fully unanimous group. So mean-only
lets the informative mixed prompts dominate the update (mass 4) while the near-unanimous ones fade
(0.9375) and the unanimous ones drop out entirely (0) — which is the weighting I actually want, because a
group where everyone got the same answer carries no relative signal. Deleting the std does not merely
remove a distortion; it restores difficulty weighting that runs the *right* way. The grpo numbers were
the wrong-way version, and this is why I expect the harder splits, not the easy one, to be where
removal shows up.

I owe myself the bridge here, because the reweighting happens on the *training* prompts (MATH level
3–5) while the numbers I read are on held-out *eval* splits, and it is easy to hand-wave across that
gap. So make it mechanical. On a training prompt that a weak 0.5B model *almost always misses*, a group
comes back `k = 1` — one lucky solve. Under grpo that lone correct sample gets the extremal advantage
`+3.75` while the fifteen failures get `−0.25`, so the update pours a huge, single-sample gradient into
reinforcing one possibly-fluky solution to a hard problem. Meanwhile a genuinely mixed training prompt,
`k = 8`, the kind where the model has a real but imperfect grip and there is consistent signal to
learn, has all its samples at `±0.97` — dominated in the batch by the `k = 1` fluke. So on exactly the
hard end of the training distribution, grpo's gradient is set by noise (rare flukes amplified `3.75/0.97
≈ 3.9×`) rather than by the steadily-learnable mixed prompts. That is the concrete route by which the
per-group std should transfer into worse *hard-eval* accuracy: the competence that carries over to AMC
and MATH-500 is built from the mixed prompts, and those are the ones the std tamps down. Mean-only
centering removes the amplification — the `k = 1` fluke drops to `+0.9375` mass `0.9375`, the `k = 8`
prompt keeps mass `4` — and lets the consistent signal lead. If that mechanism is real, the recovery
should be visible on MATH-500 and AMC and absent on GSM8K.

Cross-check the diagnosis from the other side so I have not strawmanned grpo. Its effective advantage is
`(R_i − mean(R))/std(R)`. Compared to `Ã_i = R_i − mean(R)`, grpo is the RLOO-scaled centered advantage
divided per-prompt by `std(R)`. So grpo = the correct centered advantage × a per-prompt `1/std`
reweighting. I am not proposing a different estimator; I am removing one multiplicative distortion from
the same one — and the distortion is exactly the difficulty bias the grpo numbers exposed.

I have to be honest about a piece of the full critic-free fix that I *cannot* apply here, because the
edit surface forbids it. The other classic pathology in this family is the per-response `1/|o|` in the
loss aggregation, which penalizes long wrong answers *less* per token and drives runaway length. But
that `1/|o|` lives in the actor loss's token-reduction, and the actor loss is fixed *outside* my edit —
I only get to write `compute_custom_advantage`. So I cannot replace the per-response divisor with a
constant from here; whatever the loop does with `mask.sum(-1)`, it keeps doing. The one lever I have in
this task is the std, and the derivation says removing it is the right and unbiased thing. So my dr_grpo
edit is exactly grpo minus the std: group-mean center, broadcast, mask — and nothing about loss
aggregation, because that is not mine to touch. (The fill is in the answer.)

One more thing I must not break: the degenerate groups, which the ε floor was quietly load-bearing for
in grpo. With the std removed there is no division at all, so there is no scale factor to guard. A
unanimous group is now clean by construction: all-wrong (`k = 0`) gives every `R_i = 0`, `mean = 0`,
advantage `0`; all-correct (`k = 16`) gives every `R_i = 1`, `mean = 1`, advantage `0`. Either way the
group contributes exactly nothing this step, which is right — if every sample scored the same there is
no relative signal — and the `ε` I needed in grpo's denominator is simply no longer part of the
computation. The genuine size-1 edge case stays as defensive code: a single-response group has nothing
to compare against, so I set its mean to 0 and let the raw return through, though with the fixed loop
handing me exactly 16 responses per group it never fires.

One stability worry deserves a check before I trust the deletion, because the reflexive fear is that
stripping the normalization lets advantages blow up and saturate the fixed clip. Run the arithmetic:
with `R_i ∈ {0, 1}`, the mean-only advantage is `Ã_i = R_i − mean(R)`, so `|Ã_i| ≤ max(mean, 1−mean) <
1` — every advantage is bounded in `(−1, 1)`. Compare grpo, whose z-scores I already saw reach `±3.75`
on near-unanimous groups. So the fear is backwards: it was grpo, with its per-group `1/std`, that had
the fat tail and the outsized updates; mean-only centering *tightens* the advantage range to the unit
interval and cannot produce a `3.75`-magnitude spike at all. Removing the std is, if anything, the more
conservative estimator. There is a knock-on effect worth naming, since the actor loss's clip is fixed
and I cannot touch it: the clip engages on the product of the importance ratio and the advantage, so
grpo's `±3.75` advantages would drive many more tokens into the clipped regime than mean-only's bounded
`(−1, 1)` values will. That means the fixed clip was doing quiet, unaccounted work under grpo — absorbing
those fat-tailed advantages — and under dr_grpo it will engage more gently. Whatever destabilization I
might see is not going to come from advantage scale.

So the delta from step 1 is one deletion, forced by the derivation: where grpo divided each centered
score by its group std, I stop. Reading the grpo numbers, here is what I expect and where I am unsure.
The harder splits are where the std reweighting did its damage, so MATH-500 and AMC should be where
dr_grpo recovers — I expect both to edge up from grpo's 0.2973 and 0.0934, and the aggregate
`score_mean` to move *above* grpo's −0.5166. GSM8K is mostly easy prompts whose groups are near-unanimous
either way, so I expect it to hold around grpo's 0.4668, neither helped nor hurt much by removing the
std. The move is small in the algebra — one dropped term and a `15/16` rescale — so I should not expect
a large jump; I expect a modest, directionally clean recovery concentrated on MATH and AMC. I should be
honest that this is one seed (42) and the harder splits are small — AMC at 0.0934 is a handful of
problems — so a single benchmark could twitch either way on sampling noise; the load-bearing prediction
is the *aggregate* `score_mean` clearing −0.5166, with MATH-500 and AMC as the direction I expect the
movement to concentrate. The change
is also compute-neutral: I am deleting a dict and a divide against ~337 s of rollout and update per
step, so if the wall-clock moves at all it will be generation-length variance, not the estimator, and I
should not read any speedup as meaningful. If dr_grpo clears grpo on the aggregate, the difficulty-bias
diagnosis is confirmed.

What it will *not* fix is anything driven by the length-aggregation bias I can no longer reach from this
edit surface, and anything driven by the cross-prompt scale grpo's per-group std was crudely (and
wrongly) trying to provide. That second cost is worth pricing out, because I am walking into it
deliberately. After mean-only centering, a balanced prompt (`k = 8`) yields advantages `±0.5`; a
`k = 4` prompt yields `+0.75` for its correct responses and `−0.25` for its wrong ones; a `k = 1`
prompt yields `+0.9375` and `−0.0625`; a near-unanimous `k = 15` yields `+0.0625` and `−0.9375`. So the
per-response magnitudes now range from `0.06` to `0.94` across prompts, and the batch gradient is
dominated by whichever prompts happen to have large centered scores, on no common yardstick — a prompt
whose solved-minus-typical is `0.5` and one where it is `0.06` enter the update at those raw
magnitudes. That is the deliberate price of deletion: I threw away a distortion and, with it, the one
legitimate job the std was botching, which was to put those magnitudes on a shared scale. If dr_grpo
improves but a *global* scale normalization would do better still, that is the diagnosis already waiting
for step 3.
