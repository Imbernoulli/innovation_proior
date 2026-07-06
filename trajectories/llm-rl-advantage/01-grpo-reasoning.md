The estimator is the whole point, but it plugs into a loop, and the floor I start from is the most
established critic-free thing that loop can run at all — so the pain to start from is just turning a
group of graded responses into advantages that the fixed PPO actor loss can ascend, without standing
up a value network the loop neither wants nor can fit here. Let me write down what actually hurts when
I try to RL-tune this small math model, because the choice of baseline-from-rewards is the entire game
and I do not want to fool myself about it.

PPO is actor-critic. Alongside the policy I would have to train a second network, the value function
`V_ψ`, in practice the same size as the policy — a 0.5B critic shadowing a 0.5B actor. Let me put a
number on the "brute cost" before I lean on it, because I have exactly one H200 and I should know
whether memory is really the wall. Full-parameter training of 0.5B params in bf16: the weights are
2 bytes each, so 1.0 GB; the gradients another 2 bytes, 1.0 GB; and Adam carries an fp32 master copy
plus two fp32 moments, 12 bytes per param, 6.0 GB. That is ~8 GB of resident training state for the
actor. A policy-sized critic doubles the trainable state to ~16 GB, plus its own forward/backward
activations. On a 141 GB H200 that fits with enormous room to spare. So if I am honest, "the critic is
too expensive" cannot be a memory argument in this setting — the GPU would swallow it. The cost that
actually bites is not the footprint but the coupling and, more sharply, whether the critic can even do
its job here.

That is the deeper problem, and it is *why* the critic is there at all. The reason a value function
shows up is variance reduction in the policy gradient. The estimator is
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
(ii) close to the expected return from here. It need not be a learned network. So let me lay out the
critic-free candidates and actually walk them, not wave at them. One option is a leave-one-out mean
over the other responses in a group (RLOO); another is the full group mean; another is a single global
baseline pooled across all prompts in the batch — one scalar subtracted from every response; another is
a fixed constant. Take the global-baseline option seriously for a moment, because it is tempting: I
could subtract the batch-average reward from every `r_i` and call it a baseline. But watch what it
scores. A high reward on prompt A can mean two utterly different things — "this was a good response" or
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

I want to resist the urge to be cleverer than the information allows, because uniform broadcast looks
crude and there are tempting alternatives. I could distribute the terminal reward across tokens by some
schedule — front-load it, or weight by position — or I could use the policy's own token log-probs or
entropy as a within-sequence credit proxy, handing more advantage to the tokens the model was least
sure about. Walk that second one a step: weighting by `−log π_θ(o_t)` would systematically pour extra
advantage into low-probability tokens. On a *correct* solution those are often exactly the creative
reasoning steps I want to reinforce — but on a *wrong* solution they are the noisy, malformed tokens I
want to suppress, and the scheme cannot tell the two apart because it never looks at the reward's sign
per token. Any such allocation invents a per-token signal out of the model's own confidence, encoding
whatever bias the heuristic carries, and buys me nothing: the reward-to-go is genuinely constant along
the sequence, so there is no true per-token structure to recover and no variance to be reduced by
redistributing a constant. Every non-uniform scheme adds bias with zero informational payoff. So the
uniform broadcast is not laziness; it is forced by the reward being a single terminal bit.

The `ε` deserves one careful trace, because I want to know whether it is shaping any real advantage or
only guarding a degenerate case. Take an all-wrong group, `k = 0`: every `r_i = 0`, the mean is 0, the
numerator `r_i − mean = 0`, and the std is 0. Without `ε` that is `0/0`, a NaN that would poison the
whole batch; with `ε` it is `0/ε = 0`. So the numerator is *already* zero — the well-defined answer is
an advantage of 0, no relative signal, which is exactly correct — and `ε`'s only job is to convert the
NaN into that 0. It never scales a genuine advantage; it just keeps unanimous groups from exploding.
And the singleton branch (mean 0, std 1)? Given the fixed loop — 16 samples per prompt, batch of 128
prompts, so a `(2048, response_length)` reward tensor with 128 distinct group ids each appearing
exactly 16 times — every group is size 16 and that branch never fires. It is defensive code, not a
lever. The final shaping is a broadcast I should trace so the mask is right: after standardizing I hold
a `(2048,)` vector of per-response scalars; `scores.unsqueeze(-1)` makes it `(2048, 1)`, and
multiplying by the `(2048, response_length)` `response_mask` broadcasts the scalar onto every valid
token and zeros the padding in one stroke. So a response of `L` valid tokens carries `L` identical
copies of `Â_i` and zeros thereafter — which is what "assign the whole normalized outcome to every
token" means concretely, and it is why returns can be the very same tensor: there is no separate value
target to shape, only the broadcast advantage under the mask.

Now I have to be careful about what the edit surface here actually *is*, because the established
critic-free recipe is more than an advantage formula — it also moves the KL-to-reference anchor out of
the reward and into the loss with a non-negative per-token estimator, and it dual-clips the surrogate.
In the general derivation those pieces matter: folding KL into the reward and then z-scoring it
entangles the regularizer with the advantage, so the clean version pulls KL into the loss as its own
term and the clip lives in the actor objective. But *none of that is mine to write here*. The only
editable region is `compute_custom_advantage`; the actor loss, its clipping, and the KL-loss setting
are fixed outside the edit surface and are applied by the loop after my function returns. So in this
task the method reduces to exactly the advantage half: group-mean center, divide by group std,
broadcast to tokens, mask. The KL anchor and the clip are still there — the loop supplies them — but
they are not levers I touch, and I must not write the reasoning as if I were adding them. My step-1 edit
is precisely the group z-score advantage, and nothing else. (The distilled fill is in the answer.)

Before I settle, one limit check to make sure the whole construction degenerates to something sane in
the regime where I trust it most. Push `G → ∞`: the group mean converges to the prompt's true expected
return `θ`, the group std converges to the true reward dispersion `σ`, and the z-score converges to
`(r_i − θ)/σ`, the genuinely standardized reward — a clean, unbiased-in-the-limit object, and the
self-inclusion of `r_i` in its own mean washes out as `O(1/G)`. So the estimator is not wrong in kind;
it is the right object estimated from too few samples. At `G = 16` the numerator's baseline is a
16-sample estimate of `θ` and the denominator is a 16-sample estimate of `σ`, and the squared-mass and
`3.75`-extremal computations already showed the denominator is the fragile one — small `G` is precisely
where the per-group std stops behaving like a constant and starts acting as a difficulty reweighter.
And I cannot grow the group — it is 16 by construction, and enlarging it means paying 16× the rollout
cost. So the limit does not hand me an easy fix; it sharpens the immediate question instead. At `G = 16`
is this fragile per-group scale doing more good than harm at all? If it is a net distortion, the first
thing to try is not to estimate it more carefully but to strike it out and see what the rewards do
without it.

So at step 1 the baseline is settled: per prompt, sum the per-token rewards to recover each response's
scalar score; bucket scores by group id; compute each group's mean and std; standardize each response's
score within its group; broadcast that scalar over the response's valid tokens; return it as both
advantages and returns. This is the most established critic-free estimator the loop can run, and it is
the floor I climb from. To keep the scale of the exercise in view: 128 prompts × 16 samples × 100 steps
is ~205K rollouts, and the entire learning signal is that many 0/1 verifier bits, shaped by this
function into per-token advantages. Every bit of leverage I have is in how I turn those bits into
`Â`.

Now reason about what this floor must do, because that is the entire point of running it. The
squared-mass accounting already told me the std is a per-problem reweighter, not the innocent
"advantage normalization, everybody does it" I was tempted to treat it as. When I whiten advantages
across an entire batch, dividing by one global number folds into the learning rate and changes nothing
about relative weighting. But a *per-problem* std divides different problems by different numbers, and
that 4× reweighting toward near-unanimous prompts is a difficulty distortion that exists purely because
the normalization is scoped to the group. There is a second worry riding on the same scope: the std is
estimated from only 16 samples per problem, and with outcome rewards near-unanimous groups are common,
so the scale I divide by is itself the noisiest object in the estimator — and the `3.75` extremal case
shows how a single fluke can set it. I expect this to cost accuracy in a way the benchmarks should
reveal. The harder splits (MATH-500, AMC) are where the mix of solved and unsolved is genuine, so those
are precisely the prompts whose informative signal gets tamped down relative to the near-unanimous ones
whose weight gets inflated; the transferable reasoning signal I most want to reinforce is exactly what
the std down-weights. So I expect grpo to learn — GSM8K accuracy should hold up, since its prompts are
easier and their groups behave — but to leave accuracy on the table on the harder benchmarks, and to
land at the bottom of whatever I compare it against on the aggregate `score_mean`. Whatever the precise
split, the diagnosis is already pointed at the next step: if the per-group std is a distortion rather
than a stabilizer, the cleanest test is to delete it and see whether the harder splits recover.
