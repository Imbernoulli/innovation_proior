## Research question

We have a supervised-fine-tuned language model that can already write chain-of-thought
solutions to math problems, and a cheap, reliable way to tell — after the fact — whether a
finished solution is correct: a rule checker that compares the model's final answer to the
gold answer, returning essentially one bit per response (correct / incorrect, perhaps with a
small format bonus). We want to use reinforcement learning to push the policy toward
producing more correct solutions.

The difficulty is the shape of this reward. It exists only for a *complete* response — there
is no ground-truth signal that says "this intermediate token was good." It is a single
scalar attached to the whole trajectory, effectively realized at the last token (the point
where the answer is finally emitted and can be graded). And it is sparse and strongly
prompt-dependent: for an easy prompt almost every sample is correct, for a hard prompt almost
none are, so the raw signal carries very different amounts of information per prompt.

The precise goal is a complete recipe for turning this trajectory-level correctness signal
into a stable policy-gradient update, under tight resource limits: (1) it must not require a
second policy-sized network to be trained alongside the policy; (2) it must assign credit to
the thousands of tokens in a response given a reward defined only at the end; (3) it must
keep the per-prompt contributions comparable despite wildly different correctness rates
across prompts; (4) it must regularize the policy toward the reference model without letting
that regularizer blow up or distort the learning signal; and (5) the reward must stay an
honest carrier of the correctness signal — whatever scaling or variance control is needed
should live in one well-understood place, not be smeared across the pipeline where two
normalizations could fight each other. Closing this is the problem.

## Background

After supervised fine-tuning, RL is an established way to further improve mathematical
reasoning in LLMs (Wang et al. 2023; the WizardMath line). The dominant RL recipe in the LLM
fine-tuning era, inherited from RLHF (Ouyang et al. 2022, "Training language models to
follow instructions with human feedback"), is policy-gradient on a token-level MDP: the
prompt `q` is the initial state, each generated token `o_t` is an action, the state is the
running prefix `(q, o_{<t})`, and a scalar reward is produced for the completed response.

A few load-bearing facts about this setting frame everything below.

**The reward is a trajectory-level (bandit-like) quantity.** In the LLM context the reward
model — or, here, the rule checker — scores the *whole* completed response and the score is
conventionally assigned to the last token; every earlier token receives zero immediate
reward. There is no per-token ground truth. So the return for token `t` is just the single
end-of-sequence scalar, the same for every token in the response. This is the central
structural fact: credit assignment over a long generation has to be done from one number per
trajectory.

**The policy-gradient identity.** For any objective of the form
`E[(q,o)~D] (1/|o|) Σ_t GC(q,o,t) · log π_θ(o_t | q, o_{<t})`, the gradient is
`E[(q,o)~D] (1/|o|) Σ_t GC(q,o,t) · ∇_θ log π_θ(o_t | q, o_{<t})`. The whole design space of
"how to learn from the reward" collapses to one question: what scalar coefficient `GC`
multiplies each token's score-function term `∇_θ log π_θ(o_t|·)`? Different recipes are
different choices of `GC`. Pure supervised fine-tuning maximizes
`(1/|o|) Σ_t log π_θ(o_t|q,o_{<t})` on selected data, i.e. `GC ≡ 1`.

**Baselines reduce variance in policy gradients.** The score-function estimator
`E[ R · ∇ log π ]` is unbiased but high-variance; subtracting any action-independent baseline
`b` from the return, `E[ (R − b) · ∇ log π ]`, leaves the gradient unbiased (because
`E[ b · ∇ log π ] = b ∇ E[1] = 0`) and can sharply reduce variance. The art is choosing `b`
close to the expected return without making it depend on the action.

**Correctness rates are prompt-dependent and that skews the raw signal.** A diagnostic fact
about rule-checked math RL: sample several solutions to the same prompt and the *spread* of
correctness varies enormously by prompt — near-unanimous for easy or impossible prompts, wide
for prompts at the model's frontier. A learning signal that is the raw correct/incorrect
score therefore weights prompts very unevenly and is dominated by whichever prompts happen to
have large reward spread in the batch.

**Regularizing to the reference policy.** Optimizing a learned or rule reward to convergence
over-optimizes it — the policy drifts off the manifold of fluent text and exploits the
reward's blind spots. The standard guard is a KL penalty pulling `π_θ` back toward the frozen
reference (SFT) policy `π_ref`. How to *estimate* a per-token KL from single samples matters:
the naive Monte-Carlo estimator `log(π_θ/π_ref)` of `KL[π_θ||π_ref]` is unbiased but can be
negative on a given sample and is high-variance (Schulman, "Approximating KL Divergence").
There is a one-parameter family of unbiased estimators `−log r + λ(r−1)` when
`r = π_ref/π_θ`; the `λ = 1` member cancels the leading small-step variation while remaining
guaranteed non-negative, so it is a known low-variance tool for this single-sample penalty.

## Baselines

These are the prior recipes a new method is measured against and reacts to.

**Proximal Policy Optimization (Schulman et al. 2017).** The workhorse policy-gradient
algorithm. With probability ratio `r_t(θ) = π_θ(o_t|q,o_{<t}) / π_{θ_old}(o_t|q,o_{<t})` and
an advantage estimate `A_t`, PPO maximizes the clipped surrogate

```
J_PPO(θ) = E[ (1/|o|) Σ_t min( r_t(θ) A_t, clip(r_t(θ), 1−ε, 1+ε) A_t ) ],   ε ≈ 0.2.
```

The clip makes the objective a pessimistic lower bound on the unclipped surrogate: the change
in the ratio is ignored only when it would *improve* the objective and kept when it would
worsen it, so a single minibatch cannot move the policy too far. `A_t` is computed by
Generalized Advantage Estimation (Schulman et al. 2016) from the per-step rewards `{r_{≥t}}`
and a **learned value function** `V_ψ` that estimates the expected return from each state.
**Gap:** `V_ψ` is a separate network, typically the same size as the policy, so it doubles
the memory and compute footprint of training. Worse, in the LLM setting the reward arrives
only at the last token, so fitting a `V_ψ` that is accurate at *every* intermediate token —
which GAE needs — is itself a hard, ill-posed regression. The critic is both expensive and
shaky exactly where it is leaned on.

**RLHF / InstructGPT-style PPO (Ouyang et al. 2022).** The recipe that brought PPO to LLM
fine-tuning. A reward model `r_φ(x, y)` is trained on human preference comparisons (a
log-odds ranking loss) to emit a scalar per response; the policy is then optimized with PPO
to maximize reward while staying near the SFT model. The KL regularizer is **folded into the
reward at each token**:

```
r_t = r_φ(q, o_{≤t}) − β log( π_θ(o_t|q,o_{<t}) / π_ref(o_t|q,o_{<t}) ),
```

and `V_ψ` is initialized from the reward model. **Gap:** the per-token KL-in-reward
entangles the regularizer with the reward signal — the advantage `A_t` now mixes "is this
response good" with "how far has the token drifted from the reference," and the value
function has to model that combined quantity. The reward is still a single last-token scalar
that the critic must somehow spread over the sequence.

**Rejection-sampling fine-tuning / online RFT (Yuan et al. 2023; and its online variant).**
The cheapest way to use a correctness checker: sample several outputs per prompt, keep only
the ones with the correct answer, and supervised-fine-tune on them. In gradient-coefficient
terms this is

```
GC_RFT(q, o, t) = I(o) = 1 if o's answer is correct, else 0,
```

(online RFT only differs by sampling `o` from the current policy rather than the SFT model).
**Gap:** the coefficient is a 0/1 gate. It reinforces every correct response at exactly the
same intensity and never penalizes an incorrect one — it cannot say "this correct response
was much better than that other correct one," and it throws away all information in the wrong
answers. It is filtered imitation, not graded reinforcement; there is no signed, magnitude-
sensitive learning signal.

**Process-supervised RL (Uesato et al. 2022; Lightman et al. 2023; Math-Shepherd, Wang et
al. 2023).** Instead of one reward for the whole solution, train a *process* reward model
that scores each reasoning *step*, and give the policy a reward at the end of each step; the
advantage of a token is then the (normalized) sum of rewards from the steps that follow it.
**Gap:** it needs a per-step reward signal — either expensive human step annotations or an
automatically constructed process reward model — which is far heavier than a rule that checks
only the final answer, and the step rewards introduce their own modeling error. It buys
finer credit assignment at a real cost in supervision and machinery.

## Evaluation settings

The natural yardsticks already in use:

- **In-domain math accuracy** on grade-school and competition problems — GSM8K (Cobbe et al.
  2021) and the MATH dataset (Hendrycks et al. 2021) — with chain-of-thought reasoning,
  scored by exact-match of the final answer against the gold answer.
- **Out-of-domain generalization** to math benchmarks not represented in the RL training set
  (e.g. CMATH and other held-out math tasks), to test whether the RL improvement is broad or
  benchmark-specific.
- **Distributional metrics** that separate two ways RL could help: `Maj@K` (majority vote
  over `K` samples) versus `Pass@K` (whether any of `K` samples is correct), at a fixed
  sampling temperature — `Pass@K` reflects whether the model *can* solve the problem at all,
  `Maj@K` whether its probability mass concentrates on correct answers.
- **Protocol.** Start from a fixed SFT/instruct checkpoint; sample a group of outputs per
  prompt from the current policy; score each with the rule checker / reward model; identical
  checkpoints and decoding across the compared recipes.

## Code framework

The recipe plugs into a standard on-policy LLM-RL trainer that already exists. The data
pipeline samples a batch of prompts, rolls out several responses per prompt from the current
policy, and scores each finished response with the reward source, producing a per-response
scalar. That scalar is written into a `(batch, response_length)` tensor at the last valid
token, with a `response_mask` marking valid response positions; downstream, a clipped
policy-gradient surrogate consumes a per-token advantage tensor of the same shape. The
optimizer, the rollout engine, the reference-model log-probs, and the masked reduction
utilities are all provided.

Two slots are empty. The first is how the raw per-response reward tensor is turned into the
per-token reward tensor that the advantage stage will consume. The second is how that
per-token reward becomes a per-token advantage that the surrogate multiplies into each
token's log-prob gradient. Everything method-specific lives in those two slots.

```python
import torch


def reward_to_token_reward(token_level_scores, response_mask, **kwargs):
    """Turn the raw per-response reward tensor (scalar at the last valid token)
    into the per-token reward tensor that the advantage stage consumes.

    token_level_scores: (bs, response_length)  -- reward source's scalar at last token
    response_mask:      (bs, response_length)  -- 1 on valid response tokens
    returns:            (bs, response_length)
    """
    # TODO: the reward-side transformation we will decide on.
    pass


def compute_advantage(token_level_rewards, response_mask, index, epsilon=1e-6, **kwargs):
    """Turn the per-token reward tensor into the per-token advantage tensor
    that the clipped policy-gradient surrogate will multiply into each token's
    log-prob gradient.

    token_level_rewards: (bs, response_length)
    response_mask:       (bs, response_length)
    index:               (bs,)  -- which prompt each row was sampled for
    returns advantages:  (bs, response_length)
    """
    # TODO: the credit-assignment / baseline rule we will design.
    pass


# existing on-policy LLM-RL training step the two slots plug into
def rl_step(policy, ref_logp, old_logp, batch, optimizer, kl_coeff, clip_eps):
    token_level_scores = batch["token_level_scores"]        # reward at last valid token
    response_mask = batch["response_mask"]
    index = batch["index"]                                  # prompt id per row

    token_level_rewards = reward_to_token_reward(token_level_scores, response_mask)
    advantages = compute_advantage(token_level_rewards, response_mask, index)

    logp = policy.log_prob(batch)                           # current-policy token log-probs
    ratio = torch.exp(logp - old_logp)                      # importance ratio per token
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg = -torch.min(unclipped, clipped)                     # clipped surrogate (to minimize)
    # TODO: a reference-model regularizer term added to the loss
    loss = masked_mean(pg, response_mask)
    optimizer.zero_grad(); loss.backward(); optimizer.step()


def masked_mean(x, mask, dim=-1):
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1.0)
```

The two `# TODO` slots, plus the regularizer term, are the open pieces in this trainer.
