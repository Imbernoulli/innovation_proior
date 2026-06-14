## Research question

We have an instruction-tuned language model that already produces plausible step-by-step
solutions to math problems, and we have a reward model `r_phi(q, o)` that scores a full
solution `o` to a question `q`. We want to push the policy further with reinforcement learning:
sample solutions from the model, score them, and update the model to make high-scoring
solutions more likely and low-scoring ones less likely, without drifting so far from the
starting model that it forgets how to write fluent, on-distribution text or starts gaming the
reward model.

The dominant recipe for this is an actor-critic policy-optimization method that, alongside the
policy being trained, also trains a separate **value network** the same size as the policy, to
serve as a per-token baseline for variance reduction. In the language-model setting this carries
a sharp, concrete cost. First, the value network roughly doubles the trainable footprint and
must be optimized in lockstep with the policy — a real burden when the policy is a 7B-parameter
model and memory is the binding constraint. Second, the reward signal in this setting is
extremely sparse along the sequence: the reward model emits a single scalar for the *entire*
solution, attached effectively to the last token, with nothing in between. A per-token value
function is exactly the object that is supposed to spread that terminal signal back over the
intermediate tokens, but fitting a value head that is accurate at *every* token from a single
end-of-sequence reward is hard, and a poorly-fit value function feeds biased, noisy advantages
back into the policy update.

The precise goal is an RL fine-tuning procedure that (1) keeps the stable, clipped
policy-optimization update that has made actor-critic methods the workhorse of RL-from-human-
feedback; (2) keeps a variance-reducing baseline so the policy-gradient estimate is usable;
(3) avoids the separately-trained, policy-sized value network and the per-token value-fitting
problem it creates under a last-token-only reward; and (4) keeps the policy anchored to the
starting model through a KL regularizer, so it neither collapses nor reward-hacks. Each existing
method below achieves some of these; none achieves all four at once on a single-scalar,
last-token reward without relying on a learned per-token critic.

## Background

By this time, RL fine-tuning of language models is the standard final stage after supervised
fine-tuning (SFT), and the prevailing wisdom is built on a small stack of load-bearing ideas.

The **policy gradient** is the foundation. For a stochastic policy `pi_theta`, the gradient of
the expected return has the estimator `g_hat = E[ grad_theta log pi_theta(a|s) * A_hat ]`, where
`A_hat` is an estimate of the advantage of taking action `a` in state `s`. A central, much-used
fact is that **subtracting any baseline `b(s)` that does not depend on the action leaves this
gradient unbiased** while reducing its variance, and the variance-minimizing choice of the
per-action multiplier is the advantage `A^pi(s,a) = Q^pi(s,a) - V^pi(s)` — the value of the
action relative to the state's default value. So a good baseline is, by definition, something
close to the state value `V^pi(s)`: the policy-gradient step then increases the probability of
better-than-average actions and decreases worse-than-average ones.

Concretely, the standard way to get such a baseline is **generalized advantage estimation
(GAE; Schulman et al. 2015)**. Given a learned value function `V(s)`, define the temporal-
difference residual `delta^V_t = r_t + gamma * V(s_{t+1}) - V(s_t)`; this is itself a one-step
advantage estimate, exact in expectation when `V = V^pi`. GAE forms the exponentially-weighted
average of multi-step versions, `A_hat^{GAE(gamma,lambda)}_t = sum_{l>=0} (gamma*lambda)^l *
delta^V_{t+l}`, which interpolates between the low-bias/high-variance Monte-Carlo return minus
`V` (`lambda=1`) and the low-variance/high-bias single TD residual (`lambda=0`). The whole
construction rests on having a `V(s)` that is reasonably accurate at every state/token; that is
precisely the network that is expensive to carry and hard to fit here.

A second load-bearing idea is the **importance-sampled, clipped surrogate**. When you reuse a
batch of trajectories sampled from an old policy `pi_{theta_old}` for several gradient steps,
the per-token probability ratio `rho_t = pi_theta(o_t|.) / pi_{theta_old}(o_t|.)` corrects for
the off-policyness, and the surrogate `E[ rho_t * A_t ]` has the right gradient at
`theta = theta_old` (where `rho_t = 1`). Maximizing this surrogate unconstrained, however,
produces destructively large updates, because nothing stops `rho_t` from blowing up. The
standard fix clips it: `min( rho_t * A_t, clip(rho_t, 1 - eps, 1 + eps) * A_t )`, a pessimistic
lower bound on the unclipped surrogate that removes the incentive to move `rho_t` outside
`[1-eps, 1+eps]` — it only ignores a ratio change when that change would *improve* the
objective, and counts it when it would worsen it.

A third is the **KL anchor**. Left alone, the policy will exploit imperfections in the reward
model, producing text that scores high but is degenerate or off-distribution. The established
remedy keeps the policy close to the SFT reference `pi_ref` with a KL penalty. In the standard
recipe this penalty is folded **into the reward**: the per-token reward becomes
`r_t = r_phi(q, o_{<=t}) - beta * log( pi_theta(o_t|.) / pi_ref(o_t|.) )`, so the policy pays a
running cost for diverging from the reference, with `beta` the single most important knob. A
related observation about estimating a KL `KL[q || p]` from samples of `q`: the naive single-
sample estimator `log(q(x)/p(x))` is unbiased but signed and high-variance; one can add a
control variate with zero expectation to reduce the variance, and a particular choice
`(p(x)/q(x)) - log(p(x)/q(x)) - 1` is not only unbiased but, because `log u <= u - 1`,
**always non-negative** — a per-sample KL estimate that never goes negative.

Useful **diagnostic experience** in reasoning RL separates online and offline sampling. Sampling
outputs from the *live* policy and training on them ("online") is close to sampling once from the
frozen SFT model ("offline") early in training, because the policies are still similar; as the
live policy moves, its own current errors and near-misses become the more relevant training data.
Reward models are also intrinsically *comparative* objects: they are trained on datasets of
preferences between multiple outputs for the *same* prompt, so their scores are most meaningful
relative to other responses to that same prompt rather than on any absolute scale.

## Baselines

These are the prior methods a new RL fine-tuning procedure would be measured against and react
to. A useful lens, knowable at this point, is that the gradient of essentially every one of
these can be written as
`grad_theta J = E_{(q,o) ~ D} ( (1/|o|) sum_t GC(q, o, t) * grad_theta log pi_theta(o_t|q,o_<t) )`,
i.e. a log-likelihood gradient weighted by a per-token **gradient coefficient** `GC`, differing
in the data source `D` and in how `GC` is formed from the reward signal.

**Supervised fine-tuning (SFT).** Maximize the log-likelihood of curated solutions:
`J_SFT = E_{q,o ~ P_sft} ( (1/|o|) sum_t log pi_theta(o_t|q,o_<t) )`. In the gradient-coefficient
lens, `GC_SFT = 1` for every token of every selected example. **Gap:** it can only imitate the
fixed dataset; it has no mechanism to push beyond the demonstrations or to down-weight bad
behavior, since there is no reward signal and no negative examples.

**Rejection-sampling fine-tuning (RFT; Yuan et al. 2023) and its online variant.** Sample
several outputs per question — from the SFT model (RFT) or from the live policy (Online RFT) —
keep only those whose final answer is correct, and run SFT on the survivors. The gradient
coefficient is the binary indicator `GC_RFT = I(o has correct answer) in {0, 1}`. **Gap:** the
signal is a hard 0/1 gate. It never penalizes a wrong answer (it just discards it, contributing
zero gradient), and it reinforces *every* correct answer with the same intensity regardless of
how good or how marginal it is — it cannot reward a confidently-correct solution more than a
barely-correct one, and it cannot push probability *down* on a wrong one.

**Direct preference optimization (DPO; Rafailov et al. 2023).** Skip the explicit reward model
and the sampling loop: take pairs of a preferred output `o+` and a dispreferred `o-` for the
same prompt and minimize a pairwise logistic loss whose gradient coefficient is
`GC_DPO = sigmoid( beta*log(pi_theta(o-)/pi_ref(o-)) - beta*log(pi_theta(o+)/pi_ref(o+)) )`,
raising the likelihood of `o+` and lowering that of `o-`. **Gap:** it is offline and pairwise —
the pairs are drawn once from the SFT model, so it does not learn from the *current* policy's
mistakes, and it consumes preferences as discrete pairs rather than scalar rewards on a group
of sampled outputs.

**Actor-critic clipped policy optimization (PPO; Schulman et al. 2017).** The workhorse RL-fine-
tuning method. Maximize the clipped surrogate
`J_PPO = E_{q, o ~ pi_old} ( (1/|o|) sum_t min[ rho_t * A_t, clip(rho_t, 1-eps, 1+eps) * A_t ] )`,
with `A_t` computed by GAE from a **learned value function** `V_psi`, and the KL-to-reference
penalty folded into the per-token reward as above. In the gradient-coefficient lens (taking a
single update per rollout so `rho_t = 1`), `GC_PPO = A_t`, the GAE advantage — a real-valued,
signed, magnitude-aware coefficient, which is exactly the expressiveness RFT lacks. **Gap:** it
requires the separate, policy-sized value network `V_psi` for that advantage. That network is a
heavy memory and compute cost, and in this setting it must be fit to assign accurate per-token
values from a reward that arrives only at the last token of the sequence — the regime where a
per-token value function is hardest to estimate well.

**Process-reward variants (Math-Shepherd; Wang et al. 2023).** Instead of one terminal reward,
train a process reward model that scores each intermediate reasoning *step*, and feed those
denser rewards into the same PPO machinery. **Gap:** denser per-step rewards help the credit-
assignment problem, but the method still rides on the learned-critic actor-critic stack and its
costs; the dense reward changes the signal, not the need for a value network.

## Evaluation settings

The natural yardsticks already in use for math-reasoning RL fine-tuning:

- **GSM8K** (grade-school word problems) and **MATH** (competition problems), scored by
  final-answer accuracy with chain-of-thought reasoning; these are the in-domain training/eval
  pair. Out-of-domain generalization is read off held-out math benchmarks (e.g. competition
  sets such as AMC/AIME-style problems and CMATH) that the RL phase is *not* trained on.
- Protocol: start from a fixed SFT/instruction-tuned policy; sample multiple outputs per
  training question from the (old) policy; score with a reward model whose training data comes
  from rule-based answer-correctness judgments; update the policy; evaluate periodically.
- Useful diagnostics of the *training dynamics*: comparing online vs. offline sampling, outcome
  vs. process supervision, single vs. iterative RL rounds, and Maj@K / Pass@K behavior as proxies
  for whether RL sharpens the existing output distribution or adds new capability.
- Typical knobs for the policy-optimization stage: a small policy learning rate, a KL-penalty
  coefficient, a number of sampled outputs per question, a maximum response length, and a
  training batch size; commonly a single policy update per exploration (sampling) stage.

## Code framework

The same on-policy RL fine-tuning loop already provides the surrounding machinery: a data
pipeline that yields questions; a sampler that draws responses from the policy; a reward model
that scores responses; a frozen reference policy; an optimizer; and a training loop that
alternates *generate -> score -> estimate credit -> compute the actor loss -> step*. The open
slots are the credit estimator and the actor-loss primitive.

```python
import torch


def estimate_credit(rewards, response_mask, prompt_index, **kwargs):
    """Turn the reward signal for a batch of sampled responses into a per-token
    advantage tensor of shape (batch, response_len).

    rewards:        per-token reward tensor (here nonzero only at the final token
                    of each response -- the reward model emits one scalar per response).
    response_mask:  1 for response tokens, 0 for prompt/padding.
    prompt_index:   which prompt each row belongs to, available because several
                    candidate responses may be sampled per prompt.

    TODO: fill in the credit estimator.
    """
    pass


def actor_loss(old_log_prob, log_prob, advantages, response_mask, **kwargs):
    """Given log-probs of the sampled tokens under the old (sampling) policy and the
    current policy, plus the per-token advantages, return the scalar objective to descend.

    TODO: fill in the actor update.
    """
    pass


# existing on-policy RL fine-tuning loop the two functions plug into
def train(policy, ref_policy, reward_model, optimizer, data_loader, samples_per_question):
    for questions in data_loader:
        old_policy = policy.snapshot()                      # pi_theta_old for this rollout
        responses, old_log_prob = old_policy.sample(
            questions, n=samples_per_question)
        rewards = reward_model.score(questions, responses)  # one scalar per response
        prompt_index, response_mask = index_and_mask(questions, responses)

        advantages = estimate_credit(                       # <- slot (a)
            rewards, response_mask, prompt_index)

        log_prob = policy.log_prob(questions, responses)    # under current pi_theta
        ref_log_prob = ref_policy.log_prob(questions, responses)
        loss = actor_loss(                                   # <- slot (b)
            old_log_prob, log_prob, advantages, response_mask,
            ref_log_prob=ref_log_prob)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

The loop already supplies sampled responses with their old-policy log-probs, prompt membership,
and a single terminal reward each; the two stubs are where the credit rule and actor update live.
