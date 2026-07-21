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

First let me read the table more carefully than "GSM8K held, the rest sagged." GSM8K accuracy is 0.4668
and `gsm8k_reward_mean` is also 0.4668 — identical, which pins down that the verifier reward is the
correctness indicator itself: the mean reward equals the fraction solved, so `r_i ∈ {0, 1}` for certain
now, not assumed. The three accuracies form a steep ladder 0.4668 / 0.2973 / 0.0934, unweighted mean
`0.2858` (the reported `score_mean` −0.5166 is a normalized aggregate that also folds in the hidden AIME
split, so I keep the two separate). AMC at 9.3% is near the floor of competence for a 0.5B model, so it
is the most sensitive probe of whether the update spends its gradient on the right prompts — the split I
most want to move, and the one the std should have hurt most.

Let me write down exactly what the grpo estimator does and look at it as if I had never seen it. For a
prompt I draw 16 responses, score them `R = {R_1,…,R_G}`, and set the advantage of every token of
response `i` to `(R_i − mean(R)) / (std(R) + ε)`, broadcast over its tokens. The group mean is doing
the job a value network would do — my cheap baseline, no critic to train, which is the whole reason I am
here. I believe in the baseline; GSM8K confirms it. But the `std(R)` in the denominator I have been
treating with suspicion since step 1, and I want to ask, very literally: what does that term do to the
gradient on a single token, and is there any version of it I should keep?

Before I delete anything, the honest options for that denominator are three: floor the per-group std
(clip it from below at, say, 0.25 so near-unanimous groups cannot blow up), replace it with a single
scale over a much larger pool, or delete it outright. Flooring is least disruptive but treats only the
*brittleness*: every prompt is still divided by its own number, so the difficulty reweighting from step
1 survives — a balanced group divided by ~0.52, a near-unanimous one by the floor 0.25, and the 4× tilt
toward uninformative prompts barely moves. It patches the symptom and leaves the disease. The
larger-pool option is more principled but a real redesign; I should not reach for it before establishing
that the *clean* thing — deleting the term — lands somewhere defensible. If deletion is a principled
estimator in its own right, that is the move to make first, and any scale question becomes a separate,
later one.

So does deletion leave me with an estimator I can defend, or just "grpo minus a term"? Whitening
advantages to zero mean and unit variance stabilizes training — but only when pooled wide enough to be a
constant. Whitening across an entire batch divides every advantage by one global number, which folds
into the learning rate and changes no relative weights. grpo instead divides *per prompt, per group*,
and that is the step-1 diagnosis: `Ã_i / std(R)` up-weights the almost-all-correct and almost-all-wrong
prompts (tiny std) and tamps down the mixed-outcome ones (large std). MATH and AMC are where the mix of
solved and unsolved is real, so those are the prompts whose large std tamps them down — the −0.5166
score and the depressed 0.2973 / 0.0934 are that difficulty bias made visible.

Now I want to be sure this term is not supposed to be there, so I derive the gradient and watch whether
a std ever appears. Maximizing `J = E_q E_{o~π_θ}[R(q,o)]` (dropping the KL term — a rule-based verifier
gives no reward-model distribution to stay near in the *advantage*, and the KL-loss setting is fixed
outside my edit), the policy gradient is `∇J = E[ Σ_t ∇log π_θ(o_t|·) · R(q,o) ]`; the reward-to-go
replacement and an action-independent baseline `B` (unbiased because
`E_{o_t}[ ∇log π_θ(o_t|·) B ] = B · ∇_θ 1 = 0`) give the advantage `reward-to-go − B`, whose
textbook-best `B` is the state value I am avoiding. But my reward is outcome-level, zero except the last
token, so the reward-to-go from *any* `t` is the whole return `R(q,o)`: the advantage is one scalar per
response, and the estimator reduces to "pick `B`, compute `R_i − B`, broadcast." Everything rides on
`B`, and my 16 samples' mean is a Monte-Carlo estimate of the prompt's expected return.

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
rate. So deleting the std does not leave me a biased fragment; it leaves me the unbiased RLOO estimator,
uniformly rescaled by `15/16` — a factor the learning rate cannot tell apart from itself. That converts
"delete the std because it looks bad" into "delete the std because the gradient never asked for it."

There is a sharper way to see deletion is not just harmless but *corrective*: the squared-mass view from
step 1. grpo's z-score forces every non-unanimous group to identical total mass `Σ_i z_i² = 15`, whereas
mean-only centering leaves mass `16·p(1−p)` — `4` for a balanced group, `0.9375` for a near-unanimous
one, `0` for a unanimous one. So mean-only lets the informative mixed prompts dominate and the
near-unanimous ones fade, which is the weighting I actually want. Deleting the std does not merely remove
a distortion; it restores difficulty weighting that runs the *right* way.

The reweighting happens on the *training* prompts (MATH level 3–5) while the numbers I read are on
held-out *eval* splits, so make the transfer mechanical. On a training prompt a weak 0.5B model almost
always misses, a group comes back `k = 1`; grpo hands that lone lucky solve `+3.75` and the fifteen
failures `−0.25`, pouring a single-sample gradient into a possibly-fluky solution, while a genuinely
mixed `k = 8` prompt — real but imperfect grip, consistent signal to learn — sits at `±0.97` and is
dominated by the fluke. So at the hard end of the training distribution grpo's gradient is set by noise
rather than by the steadily-learnable mixed prompts, and the competence that carries over to AMC and
MATH-500 is built from exactly those mixed prompts. Mean-only centering removes the amplification and
lets the consistent signal lead, so if the mechanism is real the recovery should show on MATH-500 and
AMC and be absent on GSM8K.

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

One stability worry, because the reflexive fear is that stripping the normalization lets advantages blow
up and saturate the fixed clip. But with `R_i ∈ {0, 1}` the mean-only advantage `Ã_i = R_i − mean(R)` is
bounded, `|Ã_i| ≤ max(mean, 1−mean) < 1`, while grpo's z-scores reach `±3.75` on near-unanimous groups.
So the fear is backwards: mean-only centering *tightens* the range to the unit interval and cannot spike,
so it is the more conservative estimator — it was grpo's fat-tailed advantages driving tokens into the
fixed clip, not these. Whatever destabilization I might see will not come from advantage scale.

The change from step 1 is one deletion: where grpo divided each centered score by its group std, I
stop. MATH-500 and AMC are where the std reweighting did its damage, so those should be where dr_grpo
recovers — I expect both to edge up from grpo's 0.2973 and 0.0934 and the aggregate `score_mean` to move
above grpo's −0.5166, while GSM8K holds around 0.4668. But the move is small in the algebra — one dropped
term and a `15/16` rescale — so I expect a modest, directionally clean recovery, not a leap; and on one
seed (42) with AMC at 0.0934 a handful of problems, a single benchmark could twitch on sampling noise,
so the load-bearing prediction is the *aggregate* clearing −0.5166. If it does, the difficulty-bias
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
