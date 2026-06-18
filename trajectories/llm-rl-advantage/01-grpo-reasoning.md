The estimator is the whole point, but it plugs into a loop, and the floor I start from is the most
established critic-free thing that loop can run at all — so the pain to start from is just turning a
group of graded responses into advantages that the fixed PPO actor loss can ascend, without standing
up a value network the loop neither wants nor can fit here. Let me write down what actually hurts when
I try to RL-tune this small math model, because the choice of baseline-from-rewards is the entire game
and I do not want to fool myself about it.

PPO is actor-critic. Alongside the policy I would have to train a second network, the value function
`V_ψ`, in practice the same size as the policy — a 0.5B critic shadowing a 0.5B actor. On one GPU that
roughly doubles the trainable footprint and forces the critic to be optimized in lockstep with the
actor. That is the brute cost. But the deeper problem is *why* the critic is there and whether it can
even do its job in this setting. The reason a value function shows up at all is variance reduction in
the policy gradient. The estimator is `g = E[ ∇_θ log π_θ(a|s) Â ]`, and the multiplier `Â` tells each
token whether to go up or down. The raw return there is unbiased but extremely noisy, so the classic
fix is to subtract a baseline `b(s)` depending only on the state, not the action — that subtraction is
free in expectation, `E_a[ ∇_θ log π_θ(a|s) b(s) ] = b(s) ∇_θ Σ_a π_θ(a|s) = b(s) ∇_θ 1 = 0`, so any
action-independent baseline leaves the gradient unbiased while it can slash the variance. The
variance-minimizing thing to put in front of `∇ log π` is the advantage `A = Q − V`: the action's value
relative to the state's default value. So the critic is not decoration — a good baseline, by
definition, approximates the state value. Concretely PPO builds advantages with GAE: given a learned
`V`, the TD residual `δ_t = r_t + γV(s_{t+1}) − V(s_t)` and `Â_t = Σ_l (γλ)^l δ_{t+l}`. Stare at that
and the dependence is glaring — every term needs an accurate `V` at every timestep.

And that is exactly what is broken here. Where does the reward live? The rule-based verifier reads the
*entire* solution and emits *one number*, placed at the last valid token, zero everywhere in between.
The value function is supposed to smear that terminal scalar backward into per-token credit, but I am
asking it to learn an accurate value at every intermediate token of a long reasoning chain from a
single end-of-sequence bit. That is a hard regression, and a badly fit `V` feeds biased, noisy
advantages straight into the policy update. So I am paying for a policy-sized network whose central job
— accurate per-token values — is precisely the job the last-token-only reward makes hardest. I want
PPO's stable clipped update, which the loop already fixes for me, and I want a variance-reducing
baseline; I do not want this learned per-token critic.

So the question sharpens: can I get a baseline — an approximation to "the default value of this state"
— without learning `V`? The baseline only has to be (i) independent of the action being scored and
(ii) close to the expected return from here. It need not be a learned network. What do I already have
lying around that is free? The thing I keep underusing: for each problem I do not sample one solution,
I sample a *group* — the loop draws 16 responses per prompt from the old policy, and each gets a
verifier score. The expected reward of the rollout policy on this problem is exactly what those 16
samples estimate. So the empirical group mean `mean(r_1,…,r_G)` *is* a Monte-Carlo estimate of the
problem's response-level value at collection time — the very quantity `V` was trying to approximate,
at the only granularity the reward actually exists (the whole response). It costs nothing extra; the
samples are already drawn. Use the group mean as the baseline.

Let me sanity-check that this is a legitimate baseline and not a trick. For a fixed problem, the
baseline I subtract from response `i` is the group mean. Is it independent of `r_i`? Not exactly — `r_i`
is one of the terms in its own mean. The cleanest REINFORCE-style argument would use a leave-one-out
mean over the other responses; the full group mean is symmetric, cheap, and with a large `G` differs
from leave-one-out only by a `1/G` self-inclusion effect. I will not pretend nothing changes; it is the
practical estimator I want, and it gives the sign structure I need — `r_i − mean(r)` is positive if this
solution beat the problem's average and negative if it trailed. There is a second reason it is the
*right* baseline and not merely cheap, and it comes from the reward itself. The verifier's signal is
inherently *comparative within a prompt*: "this solution to this problem beat the typical solution to
this problem" is far more meaningful than "this raw reward is high," which mostly tells me the problem
was easy. A per-problem, relative baseline matches the comparative nature of the signal.

Now the spread of `r_i` differs wildly across problems. An easy problem where all 16 samples are
correct has rewards bunched at the top, tiny variance; a hard one has them splayed across the range. If
I use the raw `r_i − mean(r)`, the easy problem contributes microscopic advantages and the hard one
huge ones, just because of the scale of the rewards, not because one matters more for learning. I want
each problem to contribute on a comparable footing — to be telling me "this response was *this many
standard deviations* better than typical for this problem." So normalize by the group standard
deviation too: `Â_i = (r_i − mean(r)) / (std(r) + ε)`. A per-problem z-score of the reward. This makes
the advantage scale-invariant across problems, controls the update magnitude so one high-variance
problem cannot dominate the batch, and keeps the comparative reading clean. I need the `ε` floor so a
zero-variance group — all 16 samples identical reward — does not divide by zero; and if a group somehow
has a single sample, there is no spread, so I set its mean to 0 and std to 1 and let the raw reward
through.

One thing falls out and I should be explicit. The fixed actor loss wants a genuinely *per-token*
advantage, varying along the sequence — that is what GAE produced. My group construction gives me one
scalar per *response*. With no critic and a single terminal reward, there is no information to
distinguish tokens within a correct solution; I have no per-token signal at all. The reward-to-go from
*any* position equals the whole-sequence return, because the only reward is at EOS. So the honest,
minimal choice is to assign the whole normalized outcome to every token of the response,
`Â_{i,t} = Â_i` for all `t`. Outcome supervision, broadcast across the sequence — the simplest thing
consistent with the information I actually have. The returns tensor the loop expects downstream is, with
no bootstrapped value target to compute, the same tensor as the advantages.

Now I have to be careful about what the edit surface here actually *is*, because the established
critic-free recipe is more than an advantage formula — it also moves the KL-to-reference anchor out of
the reward and into the loss with a non-negative per-token estimator, and it dual-clips the surrogate.
In the general derivation those pieces matter: folding KL into the reward and then z-scoring it
entangles the regularizer with the advantage, so the clean version pulls KL into the loss as its own
term and the clip lives in the actor objective. But *none of that is mine to write here*. The only
editable region is `compute_custom_advantage`; the actor loss, its clipping, and the KL-loss setting
are fixed outside the edit surface and are applied by the loop after my function returns. So in this
task the method reduces to exactly the advantage half: group-mean center, divide by group std, broadcast
to tokens, mask. The KL anchor and the clip are still there — the loop supplies them — but they are not
levers I touch, and I must not write the reasoning as if I were adding them. My step-1 edit is precisely
the group z-score advantage, and nothing else. (The distilled fill is in the answer.)

So at step 1 the baseline is settled: per prompt, sum the per-token rewards to recover each response's
scalar score; bucket scores by group id; compute each group's mean and std; standardize each response's
score within its group; broadcast that scalar over the response's valid tokens; return it as both
advantages and returns. This is the most established critic-free estimator the loop can run, and it is
the floor I climb from.

Now reason about what this floor must do, because that is the entire point of running it. The std in the
denominator is computed *per group, per problem* — and that is the term I am least sure about. When I
whiten advantages across an entire batch, dividing by one global number folds into the learning rate and
changes nothing about relative weighting. But a *per-problem* std divides different problems by different
numbers, and that does change their relative weight. A problem whose 16 samples are almost all correct
has tiny std; one almost all wrong also has tiny std; the mixed-outcome problems — the genuinely
informative ones — have large std. Dividing by std multiplies up the update weight of the too-easy and
too-hard problems and tamps down the mixed ones, a difficulty distortion that exists purely because the
normalization is scoped to the group. There is a second worry: the std is estimated from only 16
samples per problem, and with outcome rewards near-unanimous groups are common, so the scale I divide by
is itself the noisiest object in the estimator. I expect this to cost accuracy in two ways that the
benchmarks should reveal — the harder splits (MATH-500, AMC) are where the difficulty distortion bites
hardest, because those are exactly the prompts whose reward spread is large or whose groups are
near-unanimous. So I expect grpo to learn — GSM8K accuracy should hold up — but to leave accuracy on the
table on the harder benchmarks relative to an estimator that does not reweight by per-group spread.
Whatever the precise split, the diagnosis is already pointed at the next step: if the per-group std is a
distortion rather than a stabilizer, the cleanest test is to delete it and see whether the harder splits
recover.
