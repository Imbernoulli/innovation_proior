## Research question

After supervised fine-tuning, a language model can be pushed further by reinforcement learning
against a reward signal — a learned reward model, or a rule that checks whether a generated
answer is correct. The standard tool for this stage is Proximal Policy Optimization, an
actor-critic method. But the actor-critic recipe carries a learned value/critic network of
roughly the same size as the policy, and in the language-model setting the reward is unusual:
the reward model emits a *single scalar per response*, attached effectively at the last token,
not a dense per-step signal. The precise problem is to keep the stable, variance-reduced policy
update that makes on-policy RL work — an unbiased policy-gradient direction with a low-variance
baseline subtracted — while (1) not paying for a second policy-sized network, and (2) not
depending on a per-token value function that is hard to fit when the only reward arrives at the
end of the sequence. A solution has to produce a per-token advantage signal from the rewards
alone, stay on-policy, and remain stable under several gradient steps per batch of rollouts.

## Background

By this time RL fine-tuning of instruction-tuned LLMs is a standard third stage after
pretraining and SFT. The dominant pipeline (Ouyang et al. 2022, "Training language models to
follow instructions with human feedback") is: collect comparison data, train a reward model on
pairwise preferences over outputs *for the same prompt*, then optimize the policy with PPO
against that reward model, regularized by a KL penalty to a frozen reference (the SFT model) to
prevent the policy from drifting into degenerate, reward-hacking regions.

Several pieces of theory are load-bearing here.

**The policy-gradient identity with a baseline.** For a parameterized policy `pi_theta`, the
gradient of expected return has the score-function form `E[ sum_t grad log pi_theta(a_t|s_t) *
Psi_t ]`, where `Psi_t` is some measure of how good action `a_t` was. A central fact is that
subtracting *any* function of the state alone — a baseline `b(s)` — from `Psi_t` leaves the
gradient **unbiased**, because `E_{a~pi}[ grad log pi(a|s) * b(s) ] = b(s) * grad sum_a pi(a|s)
= b(s) * grad 1 = 0`. The baseline does not change the expected direction, only its variance,
and the variance-minimizing choice makes `Psi_t` a centered quantity — the advantage
`A(s,a) = Q(s,a) - V(s)`, the action's value relative to the state's average value. So a good
baseline is precisely an estimate of the state's expected return under the current policy.

**Generalized Advantage Estimation (Schulman et al. 2015).** GAE is the standard way to turn a
learned value function `V_psi` and a reward stream into a low-variance, low-bias advantage:
`A_t^{GAE(gamma,lambda)} = sum_l (gamma*lambda)^l * delta_{t+l}`, with TD residual
`delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`. It interpolates between the high-variance
Monte-Carlo return (`lambda=1`) and the high-bias one-step TD estimate (`lambda=0`). Every term
leans on `V_psi` being an accurate per-step value estimate.

**The reward-shape facts in LLM fine-tuning.** Two observations about the LLM setting matter.
First, the reward model is trained on
*comparisons among outputs of the same prompt* — its scores are meaningful as a relative
ordering within a prompt, and their absolute scale and offset are arbitrary and vary by prompt.
Second, in practice the reward is delivered as one scalar per response at the terminal token,
so a value function that must be accurate at every intermediate token is being asked to densify
a signal it never directly sees — a known difficulty that makes the critic both expensive and
the least reliable part of the pipeline.

## Baselines

**PPO (Schulman et al. 2017).** The workhorse. With importance ratio
`rho_t(theta) = pi_theta(a_t|s_t) / pi_{theta_old}(a_t|s_t)`, PPO maximizes the clipped surrogate
`E[ min( rho_t * A_t, clip(rho_t, 1-eps, 1+eps) * A_t ) ]`. The clip is a pessimistic
(lower-bound) surrogate: it removes the incentive to move the ratio far when doing so would
*improve* the objective, so multiple inner gradient steps per batch of rollouts stay stable.
`A_t` is computed by GAE from a **learned value function** `V_psi`. In the RLHF setting the KL
regularizer is folded *into the reward*: `r_t = r_phi(q, o_<=t) - beta * log( pi_theta(o_t|.) /
pi_ref(o_t|.) )`. **Gap:** `V_psi` is a second network of policy size — a large memory and
compute cost — and, with the reward arriving only at the last token, the per-token value
estimates it must supply are the hardest and least trustworthy quantities in the loop; the
advantage quality is bottlenecked by exactly the component that is hardest to fit here.

**Rejection-sampling fine-tuning / Online RFT (Yuan et al. 2023).** Sample several outputs per
prompt (from the SFT model for RFT, from the live policy for Online RFT), keep the ones whose
answer is correct, and do supervised MLE on those. In policy-gradient terms its per-token
gradient coefficient is the indicator `1[o is correct] in {0,1}`. It needs no critic and no
reward-model scalar. **Gap:** the coefficient is binary — it reinforces every correct output at
the same intensity regardless of how good it is, and it never *penalizes* an incorrect output
(coefficient 0, not negative). It cannot express "this correct answer is much better than that
one" or push *down* on a wrong one; all the magnitude and sign information a reward model
carries is discarded.

**DPO (Rafailov et al. 2023).** Skip online RL entirely: from preference pairs `(o+, o-)`,
optimize a closed-form pairwise loss whose per-sample coefficient is `sigma( beta*log
pi_theta(o-)/pi_ref(o-) - beta*log pi_theta(o+)/pi_ref(o+) )`. **Gap:** offline and pairwise —
the pairs are fixed up front and sampled from the SFT model, so it does not explore with the
improving policy, and it consumes preferences as pairs rather than a scalar reward over a group.

**Batch-level reward whitening (RLHF practice, Ouyang et al. 2022).** A common stabilizer is to
whiten the scalar rewards across the whole batch — subtract the batch mean, divide by the batch
std — before they enter advantage estimation. **Gap:** one mean and std for the entire batch
mixes prompts of very different difficulty and very different reward offsets; an easy prompt
where every sample is correct and a hard prompt where every sample fails are centered against a
shared statistic, so prompt-dependent reward offsets and scales can distort the signal that
later enters the policy-gradient update.

## Evaluation settings

The natural yardsticks at the time are math-reasoning benchmarks scored by answer correctness:
GSM8K (grade-school word problems), the MATH dataset and its harder subsets (e.g. MATH-500),
and competition sets such as AMC; out-of-domain checks on additional math corpora. The metric
is accuracy of the final answer (e.g. `mean@1` / `maj@k` / `pass@k`). The training side is an
on-policy loop: sample candidate responses from the current (old) policy, score each with a
reward model or a correctness rule, compute per-token advantages, and take clipped-surrogate
gradient steps with a KL-to-reference regularizer. Learning rate, KL coefficient, rollout
multiplicity, max sequence length, and batch size are the knobs.

## Code framework

The new piece slots into an existing on-policy RL trainer. Everything around it already exists:
a rollout step that samples responses from the old policy, a reward stage that turns each
response into a scalar (placed at the response's last valid token, giving a
`(batch, response_length)` reward tensor with a `response_mask`), an advantage stage, and a
clipped-surrogate policy-loss step with a KL-to-reference term. The reward tensor may carry
row metadata from the sampler. What is *not* settled is how to turn the scalar rewards into the
per-token advantage tensor the policy loss consumes — that transformation is the empty slot.

```python
import torch


def estimate_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length): scalar reward at last valid token
    response_mask: torch.Tensor,         # (bs, response_length): 1 = valid response token
    index=None,                          # optional row metadata from the rollout sampler
    epsilon: float = 1e-6,
):
    """Turn per-response scalar rewards into the per-token advantage tensor
    consumed by the clipped-surrogate policy loss."""
    scores = token_level_rewards.sum(dim=-1)   # (bs,): recover the per-response scalar
    # TODO: the advantage rule we will design.
    advantages = torch.zeros_like(token_level_rewards)
    return advantages

def policy_loss(ratio, advantages, response_mask, eps=0.2):
    # clipped surrogate (PPO-style), already in place
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    surr = torch.min(unclipped, clipped)
    return -(surr * response_mask).sum() / response_mask.sum()
```

The advantage stage is where the rule will live; the loss and rollout around it are fixed.
