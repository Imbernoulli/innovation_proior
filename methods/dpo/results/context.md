# Context: Aligning Language Models to Human Preferences

## Research question

A large language model trained by self-supervised next-token prediction on
web-scale text absorbs an enormous range of behaviors and beliefs — including
many we do not want it to reproduce. The model "knows" common programming
mistakes, widely held misconceptions, toxic registers, and rambling
non-answers, because all of those appear in its training data. The practical
problem is one of *selection and control*: out of everything the model can do,
how do we steer it toward the responses humans actually want — helpful, honest,
on-task — without retraining from scratch and without hand-writing a reward for
every situation?

The most scalable signal we have is *relative human judgment*. People find it
much easier to say "response A is better than response B" than to write the
ideal response or to assign an absolute numerical score. So the goal is: given a
fixed dataset of pairwise preferences over model completions, produce a policy
that behaves the way the preferences indicate, while staying close enough to the
original model that it keeps its fluency and does not collapse onto a few
degenerate high-reward outputs.

A solution must (1) actually optimize the thing we care about — preference
satisfaction under a closeness constraint to the base model — and (2) be simple
and stable enough to run reliably at the scale of billion-parameter models,
without a fragile multi-stage training procedure.

## Background

**Pre-training and instruction tuning.** Self-supervised LMs at scale acquire
broad zero-shot and few-shot capability (Radford 2019; Brown 2020/GPT-3;
Chowdhery 2022). Their usefulness improves substantially when fine-tuned on
instructions with human-written completions (instruction tuning; Mishra 2022;
Sanh 2022; Chung 2022; Thoppilan 2022). But supervised demonstrations are
expensive and cap out at the demonstrator's quality.

**Learning from relative preferences.** Relative judgments of quality are
cheaper to collect than expert demonstrations, and fine-tuning on preferences
has improved translation (Kreutzer 2018), summarization (Ziegler 2019; Stiennon
2020), story-telling, and instruction-following (Ouyang 2022; Bai 2022). The
standard recipe couples two ideas: a *preference model* that links a latent
scalar reward to observed pairwise choices, and *reinforcement learning* that
pushes a policy to earn high reward.

**The Bradley-Terry preference model** (Bradley & Terry 1952). Given a latent
reward r*(x, y), the probability that completion y1 is preferred to y2 for
prompt x is

  p*(y1 ≻ y2 | x) = exp(r*(x,y1)) / (exp(r*(x,y1)) + exp(r*(x,y2)))
                  = σ(r*(x,y1) − r*(x,y2)),

with σ the logistic function. Crucially the model depends only on the
*difference* of rewards, so the reward is identifiable only up to a function of
x. Plackett-Luce (Plackett 1975; Luce 1959) generalizes this to full rankings
of K items as a product of softmaxes over the remaining items at each rank;
Bradley-Terry is the K=2 case. This under-specification (a reward and the same
reward plus any g(x) explain the data equally) is a known property of these
models, and identifiability constraints are normally imposed to get guarantees
on the reward MLE (Bong & Rukhin 2022).

**The KL-constrained reward-maximization objective.** The dominant alignment
pipeline (Ziegler 2019; Stiennon 2020; Bai 2022; Ouyang 2022) optimizes

  max_π  E_{x~D, y~π(·|x)} [ r_φ(x,y) ] − β · KL( π(y|x) ‖ π_ref(y|x) ),

where π_ref is the supervised-fine-tuned model and β controls how far the policy
may drift. The KL term is not cosmetic: it keeps the policy in the region where
the learned reward is trustworthy, preserves generation diversity, and prevents
mode-collapse onto a handful of high-reward strings.

**The closed-form optimum of a KL-regularized objective.** It is a standard
result in the reward-weighted / advantage-weighted regression line (Peters &
Schaal 2007; Peng 2019) and in the distributional-control / control-as-inference
line (Korbak 2022; Go 2023; Levine 2018) that the maximizer of an
entropy/KL-regularized reward objective is an exponential tilting of the
reference distribution:

  π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β ),
  Z(x) = Σ_{y'} π_ref(y'|x) exp( r(x,y') / β ).

The catch that keeps this from being used directly is the partition function
Z(x): for language it sums over all sequences, so it is intractable. In prior
work this closed form is therefore treated as a *target to approximate* — via
weighted regression onto samples, or via importance sampling — rather than as
something one can use exactly.

**Diagnostic observations about the existing pipeline.** Two empirical facts
about the standard pipeline motivate the search for something better. First,
running RL on a learned reward is unstable and expensive: it requires a separate
reward model, an on-policy sampling loop over a multi-billion-parameter LM, and
typically a learned value baseline plus careful reward normalization (e.g. a
human-completion baseline used as a single-sample Monte-Carlo estimate of the
normalizer). Second, a *naive* preference objective that simply raises the log
probability of the preferred completion and lowers that of the dispreferred one
(unconstrained likelihood/unlikelihood) degrades the model — it produces
degenerate, repetitive text (e.g. a summary that emits "when when when …"),
because nothing bounds the likelihood *minimization*.

## Baselines

**Three-stage RLHF with PPO** (Ziegler 2019; Stiennon 2020; Ouyang 2022; Bai
2022). Stage 1: supervised fine-tune to get π_SFT. Stage 2: fit a reward model
r_φ (a copy of the SFT network with a scalar head) by Bradley-Terry MLE,
minimizing −E[ log σ(r_φ(x,y_w) − r_φ(x,y_l)) ]; rewards are usually normalized
to zero mean per prompt to reduce variance. Stage 3: maximize the KL-constrained
objective above with PPO (Schulman 2017), constructing the per-token reward
r_φ(x,y) − β(log π_θ(y|x) − log π_ref(y|x)). Gap: three coupled models, on-policy
sampling in the loop, a value/critic to estimate, sensitivity to the KL
coefficient and to reward normalization — complex and unstable to run at scale.

**REINFORCE / policy-gradient RL** (Williams 1992) and PPO (Schulman 2017) as
the stage-3 optimizer. PPO clips the probability ratio to limit per-step policy
change. Gap: high-variance gradients without a good value baseline; the baseline
itself is hard to learn well, and control-as-inference (Levine 2018) identifies
the natural baseline as a soft value function (the log-partition term) that
standard actor-critic must approximate.

**Reward-weighted / advantage-weighted regression** (Peters & Schaal 2007; Peng
2019). Convert the KL-regularized problem into supervised regression by
weighting sampled actions by exp(reward/β) (or exp(advantage/β)). This uses the
same closed-form optimum but approximates it on samples. Gap: still needs reward
values and sampling; the partition normalizer is handled only approximately.

**Preferred-FT / supervised fine-tuning on chosen** — fine-tune only on the
preferred completions y_w. Simple and stable. Gap: it ignores the dispreferred
completions entirely, so it cannot learn the *contrast* the preferences encode,
and tends not to improve much over SFT.

**Unlikelihood** (Welleck 2019) — raise log p(y_w|x) and lower log p(y_l|x)
directly, with a coefficient on the unlikelihood term. Gap: the likelihood
*minimization* is unconstrained and drives the model into degenerate text.

**Best-of-N** — sample N completions from the base/SFT policy and return the one
the learned reward model scores highest. Strong quality, decouples reward
quality from optimization. Gap: requires N forward generations per query at test
time; computationally impractical for moderate N.

## Evaluation settings

- **Controlled sentiment generation.** Prompts are 2–8 token prefixes from the
  IMDb review dataset (Maas 2011); the policy must continue with positive
  sentiment. A pre-trained sentiment classifier
  (siebert/sentiment-roberta-large-english) serves as a ground-truth reward,
  used to *generate* synthetic preference pairs (a completion with higher
  positive probability is "preferred"). Base model: GPT-2-large. Because the
  true reward is known here, one can plot the achieved-reward vs. KL-to-reference
  frontier — the natural yardstick for *how well* an algorithm trades off reward
  against staying close to π_ref.
- **Summarization.** Reddit TL;DR posts (Völske 2017) with human preferences
  collected by Stiennon (2020); a GPT-J SFT model fine-tuned on human-written
  summaries. Metric: win rate of generated summaries against the reference
  summaries, judged by an LLM evaluator, swept across sampling temperatures.
- **Single-turn dialogue.** Anthropic Helpful-Harmless dataset (Bai 2022), ~170k
  human/assistant transcripts each ending in a preferred/dispreferred response
  pair; no standard SFT model, so a reference is formed by supervised
  fine-tuning on the chosen responses. Metric: win rate against the dataset's
  preferred response, LLM-judged.
- **Out-of-distribution transfer.** Apply summarization-trained policies to
  CNN/DailyMail news articles (Nallapati 2016).
- **Protocol notes.** Win rates are computed with an LLM judge (gpt-4-0314)
  under fixed prompts, with response order randomized; the LLM judge is
  cross-checked against human raters. Reward-vs-KL frontiers are computed by
  sweeping the conservativeness knob of each method and measuring sequence-level
  KL (sum of per-token KLs) against π_ref.

## Code framework

The primitives that already exist before the method: a causal-LM that maps token
ids to per-position logits, a tokenizer, an optimizer (RMSprop / Adam), and a
preference dataset yielding, per example, a prompt with a preferred and a
dispreferred completion. What is missing is the training objective that turns
preference pairs into a gradient on the policy.

```python
import torch
import torch.nn.functional as F

# --- given primitives ---
policy_model     = load_causal_lm()          # trainable π_θ, returns logits
reference_model  = load_causal_lm().eval()   # frozen anchor π_ref
for p in reference_model.parameters():
    p.requires_grad_(False)
optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=1e-6)

def sequence_logprob(model, input_ids, loss_mask):
    """Sum the per-token log p(token) over the completion (masked) positions."""
    logits = model(input_ids).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    per_token = torch.gather(logits.log_softmax(-1), 2,
                             labels.unsqueeze(2)).squeeze(2)
    return (per_token * loss_mask[:, 1:]).sum(-1)   # shape (B,)

def preference_objective(policy_chosen_lp, policy_rejected_lp,
                         ref_chosen_lp, ref_rejected_lp, beta):
    """The contribution: a single-stage loss on π_θ from preference pairs.

    No reward model, no value function, no on-policy sampling. Map each
    completion to an implicit score, contrast preferred vs dispreferred.
    """
    # TODO: define the implicit per-completion score from π_θ and π_ref
    # TODO: form the preferred-minus-dispreferred contrast
    # TODO: turn the contrast into a binary-cross-entropy / logistic loss
    pass

def train_step(batch, beta):
    # policy log-probs (with grad) for chosen and rejected
    pc = sequence_logprob(policy_model,    batch["chosen_ids"],   batch["chosen_mask"])
    pr = sequence_logprob(policy_model,    batch["rejected_ids"], batch["rejected_mask"])
    with torch.no_grad():  # reference is frozen
        rc = sequence_logprob(reference_model, batch["chosen_ids"],   batch["chosen_mask"])
        rr = sequence_logprob(reference_model, batch["rejected_ids"], batch["rejected_mask"])
    loss = preference_objective(pc, pr, rc, rr, beta)   # TODO returns scalar
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss
```

The single empty slot is `preference_objective`: the map from a preference pair
to a scalar loss on the policy. Everything around it — the LM forward pass, the
masked sequence log-probability, the frozen reference, the optimizer loop — is
standard and already on the table.
