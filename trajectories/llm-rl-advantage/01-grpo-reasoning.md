The estimator is the whole point, but it plugs into a loop, and the floor I start from is the most
established critic-free thing that loop can run at all — so the pain to start from is just turning a
group of graded responses into advantages that the fixed PPO actor loss can ascend, without standing
up a value network the loop neither wants nor can fit here. Let me write down what actually hurts when
I try to RL-tune this small math model, because the choice of baseline-from-rewards is the entire game
and I do not want to fool myself about it.

PPO is actor-critic. Alongside the policy I would have to train a second network, the value function
`V_ψ`, in practice the same size as the policy — a 0.5B critic shadowing a 0.5B actor. On one H200 the
footprint is not the wall: actor plus a policy-sized critic is ~16 GB of training state against 141 GB
of memory, so "the critic is too expensive" cannot be a memory argument here. What actually bites is
the coupling, and more sharply whether the critic can even do its job — which means asking *why* the
critic is there at all.

The reason a value function shows up is variance reduction in the policy gradient. The estimator is
`g = E[ ∇_θ log π_θ(a|s) Â ]`, and the multiplier `Â` tells each token whether to go up or down. The
raw return there is unbiased but extremely noisy, so the classic fix is to subtract a baseline `b(s)`
depending only on the state, not the action — that subtraction is free in expectation,
`E_a[ ∇_θ log π_θ(a|s) b(s) ] = b(s) ∇_θ Σ_a π_θ(a|s) = b(s) ∇_θ 1 = 0`, so any action-independent
baseline leaves the gradient unbiased while it can slash the variance. The variance-minimizing thing
to put in front of `∇ log π` is the advantage `A = Q − V`: the action's value relative to the state's
default value. So the critic is not decoration — a good baseline, by definition, approximates the state
value. Concretely PPO builds advantages with GAE: given a learned `V`, the TD residual
`δ_t = r_t + γV(s_{t+1}) − V(s_t)` and `Â_t = Σ_l (γλ)^l δ_{t+l}`. Stare at that and the dependence is
glaring — every term needs an accurate `V` at every timestep.

And that is exactly what is broken here. Where does the reward live? The rule-based verifier reads the
*entire* solution and emits *one number*, placed at the last valid token, zero everywhere in between.
The value function is supposed to smear that terminal scalar backward into per-token credit, but I am
asking it to learn an accurate value at every intermediate token of a long reasoning chain from a
single end-of-sequence bit. That is a hard regression, and a badly fit `V` feeds biased, noisy
advantages straight into the policy update. So I am paying for a policy-sized network — which the H200
could hold — whose central job, accurate per-token values, is precisely the job the last-token-only
reward makes hardest. I want PPO's stable clipped update, which the loop already fixes for me, and I
want a variance-reducing baseline; I do not want this learned per-token critic.

So the question sharpens: can I get a baseline — an approximation to "the default value of this state"
— without learning `V`? The baseline only has to be (i) independent of the action being scored and
(ii) close to the expected return from here. It need not be a learned network. The critic-free
candidates are a leave-one-out mean over the other responses in a group (RLOO), the full group mean, a
single global baseline pooled across all prompts in the batch, or a fixed constant. Take the global
baseline first, because it is tempting: subtract the batch-average reward from every `r_i`. But watch
what it scores. A high reward on prompt A can mean two utterly different things — "this was a good response" or
"A was an easy problem." A response that is correct on a trivial prompt where every sample is already
correct has `r_i = 1`, beats the global mean, and gets a positive advantage — reinforced, though there
was nothing to learn. Meanwhile a strong-but-imperfect response on a hard prompt can trail the global
mean and be pushed *down*. A global baseline reads difficulty as quality: it rewards easy-prompt
correctness and punishes hard-prompt effort. That is disqualifying, and it tells me the baseline must
be *per prompt*, not global. So the candidates collapse to the two per-prompt ones: leave-one-out
versus the full group mean.

Here the thing I keep underusing pays off: for each problem I do not sample one solution, I sample a
*group* — the loop draws 16 responses per prompt from the old policy, and each gets a verifier score.
The expected reward of the rollout policy on this problem is exactly what those 16 samples estimate. So
the empirical group mean `mean(r_1,…,r_G)` *is* a Monte-Carlo estimate of the problem's response-level
value at collection time — the very quantity `V` was trying to approximate, at the only granularity the
reward actually exists (the whole response). It costs nothing extra; the samples are already drawn.

Let me sanity-check that this is a legitimate baseline and not a trick, and settle the leave-one-out
versus group-mean choice with algebra rather than taste. For a fixed problem, the baseline I subtract
from response `i` is the group mean. Is it independent of `r_i`? Not exactly — `r_i` is one of the
terms in its own mean. The cleanest REINFORCE-style construction would use a leave-one-out mean over
the other `G−1` responses, `b_i = (1/(G−1)) Σ_{j≠i} r_j`, which by construction cannot depend on
`o_i`. How far apart are the two? Scale the group-mean-centered score by `G/(G−1)`:
`(G/(G−1))(r_i − mean) = r_i − (1/(G−1)) Σ_{j≠i} r_j`, which is exactly the leave-one-out advantage.
For `G = 16` the constant is `16/15 = 1.0667` — a global 6.7% rescale that folds straight into the
learning rate and changes nothing about relative weighting. So group-mean centering and leave-one-out
are the same estimator up to a constant; the group mean is symmetric and cheap, so I take it and keep
the sign structure I need — `r_i − mean(r)` is positive if this solution beat the problem's average and
negative if it trailed. There is a second reason it is the *right* baseline and not merely cheap, and
it comes from the reward itself. The verifier's signal is inherently *comparative within a prompt*:
"this solution to this problem beat the typical solution to this problem" is far more meaningful than
"this raw reward is high," which mostly tells me the problem was easy. A per-problem, relative baseline
matches the comparative nature of the signal — and it is exactly what disqualified the global baseline
a moment ago.

Now the spread of `r_i` differs wildly across problems. An easy problem where all 16 samples are
correct has rewards bunched at the top, tiny variance; a hard one has them splayed across the range. If
I use the raw `r_i − mean(r)`, the easy problem contributes microscopic advantages and the hard one
huge ones, just because of the scale of the rewards, not because one matters more for learning. I want
each problem to contribute on a comparable footing — to be telling me "this response was *this many
standard deviations* better than typical for this problem." So normalize by the group standard
deviation too: `Â_i = (r_i − mean(r)) / (std(r) + ε)`. A per-problem z-score of the reward. This makes
the advantage scale-invariant across problems and keeps the comparative reading clean. I need the `ε`
floor so a zero-variance group — all 16 samples identical reward — does not divide by zero; and if a
group somehow has a single sample, there is no spread, so I set its mean to 0 and std to 1 and let the
raw reward through.

Before I commit to that denominator I want to see what it actually does to a concrete gradient, on the
rewards this task really has. The verifier is a correctness check, so `r_i ∈ {0, 1}`, and the code uses
`torch.std`, the sample standard deviation with divisor `G−1 = 15`. Take a balanced group, 8 of 16
correct: mean `0.5`, sample std `sqrt((1/15)·16·0.25) = 0.5164`, so a correct response gets advantage
`0.5/0.5164 = 0.968` and a wrong one `−0.968`. Now a near-unanimous group where one lucky sample solved
a hard prompt, `k = 1`: mean `0.0625`, sample std `0.25`, so the lone correct response gets
`0.9375/0.25 = 3.75` and each of the fifteen wrong ones gets `−0.25`. The mirror image, `k = 15` with
one miss, hands the lone wrong response `−3.75` and the fifteen correct ones `+0.25`. So a single
rare-outcome sample in a near-unanimous group receives a z of `±3.75` — nearly four times the magnitude
of any sample in the balanced group. That `3.75` is not an accident of the example: it is the extremal
value `(G−1)/√G = 15/4` that a sample-std z-score can reach in a 16-response group, hit exactly when one
sample stands alone against the other fifteen. The denominator is silently handing enormous advantages
to the flukes.

That per-sample view is suggestive, but the honest question is what the std does to each *prompt's*
total pull on the update, so let me sum the squared advantages over a group as a proxy for how much
gradient mass that prompt claims. For the z-score this is remarkable: standardizing forces the sample
variance of the advantages to 1 with zero mean, so `Σ_i z_i² = G−1 = 15` for *every* non-unanimous
group, regardless of difficulty. The std makes every prompt contribute the identical total gradient
mass, 15. Contrast the un-normalized centered score: `Σ_i (r_i − mean)² = (G−1)·samplevar = 16·p(1−p)`,
which is `4` for a balanced group (`p = 0.5`), `3` for `k = 4`, and only `0.9375` for a near-unanimous
`k = 1` or `k = 15` group. Without the std the mixed prompts naturally dominate (mass 4) and the
near-unanimous ones fade (mass 0.9375) — which is exactly right, a near-unanimous group carries almost
no relative signal and should barely move the policy. Dividing by std overturns that ordering: it lifts
the near-unanimous prompt from 0.9375 to 15 (a 16× up-weight) and the balanced prompt from 4 to 15
(3.75×), so relative to plain centering the std reweights the batch *toward* the least informative
prompts by a factor of about `16/3.75 ≈ 4.3×`. That is the difficulty distortion, and it is not a vague
unease — it is roughly 4× on the nose, and it comes purely from scoping the normalization to a
16-sample group instead of pooling it wide enough to be a constant.

One thing falls out and I should be explicit. The fixed actor loss wants a genuinely *per-token*
advantage, varying along the sequence — that is what GAE produced. My group construction gives me one
scalar per *response*. With no critic and a single terminal reward, there is no information to
distinguish tokens within a correct solution; I have no per-token signal at all. The reward-to-go from
*any* position equals the whole-sequence return, because the only reward is at EOS. So the honest,
minimal choice is to assign the whole normalized outcome to every token of the response,
`Â_{i,t} = Â_i` for all `t`. Outcome supervision, broadcast across the sequence — the simplest thing
consistent with the information I actually have. The returns tensor the loop expects downstream is,
with no bootstrapped value target to compute, the same tensor as the advantages.

Uniform broadcast looks crude, and there are tempting alternatives — front-load the terminal reward,
weight by position, or use the policy's own token log-probs as a within-sequence credit proxy. But
weighting by `−log π_θ(o_t)` pours extra advantage into low-probability tokens, which on a *correct*
solution are the creative reasoning steps I want to reinforce but on a *wrong* one are the noisy,
malformed tokens I want to suppress — and the scheme cannot tell the two apart, because it never looks
at the reward's sign per token. Any such allocation invents a per-token signal out of the model's own
confidence and buys me nothing: the reward-to-go is genuinely constant along the sequence, so there is
no per-token structure to recover. Uniform broadcast is not laziness; it is forced by the reward being
a single terminal bit.

The `ε` floor only guards a degenerate case. An all-wrong group (`k = 0`) has every `r_i = 0`, mean 0,
numerator `r_i − mean = 0`, and std 0, so without `ε` it is `0/0`, a NaN that would poison the whole
batch; with `ε` it is `0/ε = 0`, which is the correct answer anyway — no relative signal. So `ε` never
scales a genuine advantage; it just keeps unanimous groups from exploding. The singleton branch (mean
0, std 1) is defensive code that never fires: the fixed loop hands exactly 16 responses per group, a
`(2048, response_length)` reward tensor with 128 group ids each appearing 16 times. The final
`scores.unsqueeze(-1) * response_mask` broadcasts the per-response scalar onto every valid token and
zeros padding in one stroke, and returns can be the very same tensor because there is no separate value
target to shape.

The established critic-free recipe is more than an advantage formula — it also moves the KL-to-reference
anchor out of the reward and into the loss and dual-clips the surrogate. But none of that is mine to
write here: the only editable region is `compute_custom_advantage`; the actor loss, its clipping, and
the KL-loss setting are fixed outside the edit surface and applied by the loop after my function
returns. So the method reduces to exactly the advantage half — group-mean center, divide by group std,
broadcast to tokens, mask — and I must not write it as if I were adding the KL term or the clip. (The
fill is in the answer.)

The construction is right in kind — as `G → ∞` the group mean converges to the prompt's true expected
return `θ`, the std to the true dispersion `σ`, and the z-score to the genuinely standardized reward
`(r_i − θ)/σ`. But at `G = 16` both the baseline and the fragile denominator are 16-sample estimates,
and I cannot grow the group without paying 16× the rollout cost. So the question is whether this
per-group scale does more good than harm at `G = 16`; if it is a net distortion, the first thing to try
is not to estimate it more carefully but to strike it out and see what the rewards do without it.

So step 1 is settled: sum the per-token rewards to each response's scalar score, bucket by group id,
compute each group's mean and std, standardize each score within its group, broadcast that scalar over
the response's valid tokens, return it as both advantages and returns. This is the most established
critic-free estimator the loop can run, the floor I climb from.

What this floor must do follows from the squared-mass accounting: the per-group std reweights the batch
~4× toward the near-unanimous prompts and away from the mixed-outcome ones, and it is estimated from
only 16 samples, so with outcome rewards making near-unanimous groups common, the scale I divide by is
the noisiest object in the estimator. The harder splits (MATH-500, AMC) are where the mix of solved and
unsolved is genuine, so those are precisely the prompts whose informative signal gets tamped down
relative to the near-unanimous ones whose weight gets inflated — exactly the transferable reasoning
signal I most want to reinforce. So I expect grpo to learn — GSM8K should hold, its prompts easier and
their groups near-unanimous either way — but to leave accuracy on the table on the harder benchmarks,
and to land at the bottom of whatever I compare it against on the aggregate `score_mean`. If the
per-group std is a distortion rather than a stabilizer, the cleanest test is to delete it and see
whether the harder splits recover.
