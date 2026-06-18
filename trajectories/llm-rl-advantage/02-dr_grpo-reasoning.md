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

Let me write down exactly what the grpo estimator does and look at it as if I had never seen it. For a
prompt I draw 16 responses, score them `R = {R_1,…,R_G}`, and set the advantage of every token of
response `i` to `(R_i − mean(R)) / (std(R) + ε)`, broadcast over its tokens. The group mean is doing
the job a value network would do — my cheap baseline, no critic to train, which is the whole reason I am
here. I believe in the baseline; GSM8K confirms it. But the `std(R)` in the denominator I have been
treating as "advantage normalization, everybody does it," and I want to ask, very literally: what does
that term do to the gradient on a single token?

The reading I carry around is "whitening advantages to zero mean and unit variance stabilizes
training." That is true — but the scope is where it goes wrong. When you whiten advantages across an
entire batch, you divide every advantage by one global number, the batch std, and that is just a
uniform rescale of the whole gradient — it folds into the learning rate and changes nothing about the
relative weighting of examples. But grpo computes the std *per prompt, per group*. Each prompt's
centered reward is divided by *that prompt's* std. So different prompts are divided by different
numbers, and that does change their relative weight in the update.

Make it concrete. Pull out the un-normalized centered score `Ã_i = R_i − mean(R)`, and read grpo as a
reweighting of it: the grpo advantage is `Ã_i / std(R)`, the centered score scaled per-prompt by
`1/std(R)`. Now think about which prompts have small std. A prompt whose 16 responses are almost all
correct (rewards near 1) has tiny std; one almost all wrong (rewards near 0) also has tiny std; a prompt
with a genuine mix — the informative ones, where the model sometimes gets it — has large std. So
dividing by std multiplies *up* the update weight of the too-easy and too-hard prompts and tamps *down*
the mixed-outcome ones. That is backwards from what I want, and regardless of direction it is a
distortion: the per-prompt `1/std` silently decides which prompts dominate the gradient based on nothing
but the spread of their reward, not on how much I should learn from them. A difficulty bias, existing
purely because the normalization is scoped to the group instead of the batch. And this is exactly why
the *harder* splits suffered most in the grpo numbers: MATH and AMC are where the mix of solved and
unsolved is real, so those are precisely the prompts whose large std gets them tamped down, while the
trivially-easy and trivially-hard prompts (more common on GSM8K) get their weight inflated. The −0.5166
score and the depressed 0.2973 / 0.0934 are the difficulty bias made visible.

Now I want to be sure this term is not supposed to be there. Where does dividing by a *group* std even
come from? It is a graft of two ideas: the group mean as baseline, which is principled, and
advantage-whitening, which is principled *only* when pooled large enough to be a constant. Scoping the
whiten to a 16-sample group fuses them in a way that is neither — the std is both a per-prompt reweighter
and a brittle small-sample estimate. So "delete the std" is the obvious move, but "delete it because it
looks bad" is not enough; I deleted it on a gradient hunch and I need to check that what is *left* is
principled, that removing it lands me on an estimator I can defend as unbiased, not just on "grpo minus
a term."

So derive the gradient from scratch. I am maximizing `J(π_θ) = E_q E_{o~π_θ}[R(q,o)]`. (I drop any
KL-to-reference term from the advantage reasoning — with a rule-based verifier there is no
reward-model distribution I must stay near in the *advantage*, and in any case the KL-loss setting is
fixed outside my edit.) The Monte-Carlo policy gradient is `∇J = E[ ∇log π_θ(o|q) · R(q,o) ]`. The
log-prob factorizes over tokens, so `∇J = E[ Σ_t ∇log π_θ(o_t|·) · R(q,o) ]`. Now the standard
sharpening: a token at position `t` cannot influence rewards before `t`, so I may replace `R(q,o)`
multiplying the `t`-th score by the reward-to-go `Σ_{t'≥t} r(q,o_{≤t'})` without changing the
expectation. Then the baseline: subtract `B(q,o_{<t})` to cut variance; as long as `B` does not depend
on the action `o_t`, `E_{o_t}[ ∇log π_θ(o_t|·) B ] = B · ∇_θ Σ_{o_t} π_θ = B · ∇_θ 1 = 0`, so it is
unbiased. The thing in parentheses, `reward-to-go − B`, is the advantage; the textbook-best `B` is the
expected reward-to-go, the state value — exactly the critic I am avoiding.

Now use the structure of *my* reward. It is outcome-level: zero everywhere except the last token. So
the reward-to-go from *any* position `t` is the same — the whole-trajectory return `R(q,o)`, for every
`t`. That collapses everything: the advantage is one scalar per response, broadcast across its tokens.
There is no per-token credit because there is no per-token reward. The entire estimator reduces to
"pick a baseline `B`, compute `R_i − B`, broadcast." Everything is in the choice of `B`. I do not have
a value network; I have 16 samples from the same prompt, whose mean is a Monte-Carlo estimate of the
prompt's expected return. Center by that mean: `Ã_i = R_i − mean(R)`, constant in `t`. No std has
appeared. No length term has appeared.

But I must check this centered baseline before I call it unbiased: the mean includes `R_i` itself, so it
is not action-independent in the literal proof sense. The clean multi-sample baseline is leave-one-out,
`B_i = (1/(G−1)) Σ_{j≠i} R_j`, which by construction does not depend on `o_i`. How far is the
group-mean-centered advantage from leave-one-out? Scale the centered advantage by `G/(G−1)`:
`(G/(G−1))(R_i − (1/G)Σ_j R_j) = R_i − (1/(G−1))Σ_{j≠i} R_j`, which *is* the leave-one-out advantage.
So `(G/(G−1)) · (R_i − mean(R)) = R_i − mean_{j≠i}(R_j)` exactly: the mean-only centered advantage is
the unbiased RLOO policy gradient up to the global constant `(G−1)/G` that folds into the learning rate.
That is the confirmation I wanted — deleting the std does not leave me with a biased fragment; it leaves
me with the unbiased leave-one-out estimator, rescaled.

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

One more thing I must not break: the unanimous-group case. With the std removed there is no singular
scale factor when a group is all-correct or all-wrong; `Ã_i = R_i − mean(R)` simply becomes zero for a
unanimous group, which is exactly right — if every sample scored the same, there is no relative signal
and the prompt contributes nothing this step. The degenerate case fixes itself once the std is gone, and
the `ε` I needed in grpo's denominator is no longer load-bearing. The genuine size-1 edge case stays: a
single-response group has nothing to compare against, so I set its mean to 0 and let the raw return
through.

So the delta from step 1 is one deletion, forced by the derivation: where grpo divided each centered
score by its group std, I stop. Reading the grpo numbers, here is what I expect and where I am unsure.
The harder splits are where the std reweighting did its damage, so MATH-500 and AMC should be where
dr_grpo recovers — I expect both to edge up from grpo's 0.2973 and 0.0934, and the aggregate
`score_mean` to move *above* grpo's −0.5166. GSM8K is mostly easy prompts whose groups are
near-unanimous either way, so I expect it to hold around grpo's 0.4668, neither helped nor hurt much by
removing the std. If dr_grpo clears grpo on the aggregate, the difficulty-bias diagnosis is confirmed.
What it will *not* fix is anything driven by the length-aggregation bias I can no longer reach from this
edit surface, and anything driven by the cross-prompt scale grpo's per-group std was crudely (and
wrongly) trying to provide — different prompts' advantages are now on incomparable raw scales again. If
dr_grpo improves but a *global* scale normalization would do better still, that is the diagnosis already
waiting for step 3.
